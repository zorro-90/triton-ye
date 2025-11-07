import functools
import os
import subprocess
import triton
import re
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['cuda']
PyCUtensorMap = None


@functools.lru_cache()
def libcuda_dirs():
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        mod = compile_module_from_src(
            src=Path(os.path.join(dirname, "driver.c")).read_text(),
            name="cuda_utils",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )
        global PyCUtensorMap
        PyCUtensorMap = mod.PyCUtensorMap
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = mod.set_printf_fifo_size
        self.fill_tma_descriptor = mod.fill_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


FLOAT_STORAGE_TYPE = {
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "fp32": "uint32_t",
    "f32": "uint32_t",
    "fp64": "uint64_t",
}
FLOAT_PACK_FUNCTION = {
    "fp16": "pack_fp16",
    "bf16": "pack_bf16",
    "fp32": "pack_fp32",
    "f32": "pack_fp32",
    "fp64": "pack_fp64",
}

_BASE_ARGS_FORMAT = "iiiKKppOOOOOO"
_BASE_ARGS_FORMAT_LEN = len(_BASE_ARGS_FORMAT)


def make_launcher(constants, signature, tensordesc_meta):

    def _expand_signature(signature):
        output = []
        tensordesc_idx = 0
        # Expand tensor descriptor arguments into either nvTmaDesc, shape and
        # strides, or base pointer, shape and strides depending on whether the
        # kernel was lowered to use the nvTmaDesc or not.
        for sig in signature:
            if isinstance(sig, str) and sig.startswith("tensordesc"):
                meta = tensordesc_meta[tensordesc_idx] if tensordesc_meta else None
                tensordesc_idx += 1

                match = re.match("tensordesc<([^[>]*)\\[([^]]*)\\]", sig)
                dtype = match.group(1)
                shape = match.group(2)
                ndim = shape.count(",") + 1

                if meta is None:
                    output.append("*" + dtype)
                    # Currently the host side tensor descriptors get passed in as a
                    # tensor desc, shape, and strides. We have no way to use these
                    # shape and strides when processing tensor descriptors which is
                    # why we provide our own decomposition above. Sadly this means
                    # we have to pass the shape and strides twice.
                    for _ in range(2 * ndim):
                        output.append("i64")
                    output.append("i1")
                else:
                    output.append("nvTmaDesc")

                for _ in range(ndim):
                    output.append("i32")
                for _ in range(ndim):
                    output.append("i64")
            else:
                output.append(sig)

        assert not tensordesc_meta or tensordesc_idx == len(tensordesc_meta)
        return output

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    def _extracted_type(ty):
        if isinstance(ty, tuple):
            val = ','.join(map(_extracted_type, ty))
            return f"[{val}]"
        if ty[0] == '*':
            return "PyObject*"
        if ty in ("constexpr", "nvTmaDesc"):
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        if isinstance(ty, tuple):
            val = ''.join(map(format_of, ty))
            return f"({val})"
        if ty[0] == '*':
            return "O"
        if ty in ("constexpr", "nvTmaDesc"):
            return "O"
        if ty.startswith("tensordesc"):
            return "O"
        return {
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "L",
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty_to_cpp(ty)]

    expand_signature = _expand_signature(signature.values())
    signature = {i: s for i, s in enumerate(expand_signature)}

    args_format = ''.join([format_of(ty) for ty in signature.values()])
    format = _BASE_ARGS_FORMAT + args_format

    flat_signature = []
    for sig in signature.values():
        _flatten_signature(sig, flat_signature)
    signature = {i: s for i, s in enumerate(flat_signature)}
    args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decl_list = []
    for i, ty in signature.items():
        if ty == "constexpr":
            continue
        if ty in FLOAT_STORAGE_TYPE:
            arg_decl_list.append(f"{FLOAT_STORAGE_TYPE[ty]} arg{i}")
        else:
            arg_decl_list.append(f"{ty_to_cpp(ty)} arg{i}")
    arg_decls = ', '.join(arg_decl_list)
    internal_args_list = []
    for i, ty in signature.items():
        if ty[0] == "*":
            internal_args_list.append(f"ptr_info{i}.dev_ptr")
        elif ty in FLOAT_STORAGE_TYPE:
            internal_args_list.append(f"_arg{i}_storage")
        elif ty == "nvTmaDesc":
            # Note: we have to dereference the pointer
            internal_args_list.append(f"*tma_ptr{i}")
        elif ty != "constexpr":
            internal_args_list.append(f"_arg{i}")
    params = range(len(signature))

    # generate glue code
    newline = '\n  '
    ptr_decls = [
        f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
        for i, ty in signature.items()
        if ty[0] == "*"
    ]
    tma_decls = [
        f"CUtensorMap* tma_ptr{i} = getTmaDesc(_arg{i}); if (!tma_ptr{i}) return NULL;" for i, ty in signature.items()
        if ty == "nvTmaDesc"
    ]
    float_storage_decls = [
        f"{FLOAT_STORAGE_TYPE[ty]} _arg{i}_storage = {FLOAT_PACK_FUNCTION[ty]}(_arg{i});"
        for i, ty in signature.items()
        if ty in FLOAT_STORAGE_TYPE
    ]
    params = [f"&arg{i}" for i, ty in signature.items() if ty != "constexpr"]
    params.append("&global_scratch")
    params.append("&profile_scratch")
    src = f"""
#include \"cuda.h\"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdlib.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {{
  PyObject_HEAD;
  _Alignas(128) CUtensorMap tensorMap;
}} PyCUtensorMapObject;

static inline void gpuAssert(CUresult code, const char *file, int line)
{{
   if (code != CUDA_SUCCESS)
   {{
      const char* prefix = "Triton Error [CUDA]: ";
      const char* str;
      cuGetErrorString(code, &str);
      char err[1024] = {{0}};
      strcat(err, prefix);
      strcat(err, str);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, err);
      PyGILState_Release(gil_state);
   }}
}}

#define CUDA_CHECK(ans) {{ gpuAssert((ans), __FILE__, __LINE__); }}

typedef CUresult (*cuLaunchKernelEx_t)(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra);

static cuLaunchKernelEx_t getLaunchKernelExHandle() {{
  // Open the shared library
  void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!handle) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to open libcuda.so.1");
    return NULL;
  }}
  // Clear any existing error
  dlerror();
  cuLaunchKernelEx_t cuLaunchKernelExHandle = (cuLaunchKernelEx_t)dlsym(handle, "cuLaunchKernelEx");
  // Check for errors
  const char *dlsym_error = dlerror();
  if (dlsym_error) {{
    PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve cuLaunchKernelEx from libcuda.so.1");
    return NULL;
  }}
  return cuLaunchKernelExHandle;
}}

static void _launch(int gridX, int gridY, int gridZ, int num_warps, int num_ctas, int launch_cooperative_grid, int launch_pdl, int shared_memory, CUstream stream, CUfunction function, CUdeviceptr global_scratch, CUdeviceptr profile_scratch{', ' + arg_decls if len(arg_decls) > 0 else ''}) {{
  void *params[] = {{ {', '.join(params)} }};
  if (gridX*gridY*gridZ > 0) {{
    // 4 attributes that we can currently pass maximum
    CUlaunchAttribute launchAttr[4];
    static cuLaunchKernelEx_t cuLaunchKernelExHandle = NULL;
    if (cuLaunchKernelExHandle == NULL) {{
      cuLaunchKernelExHandle = getLaunchKernelExHandle();
    }}
    CUlaunchConfig config;
    config.gridDimX = gridX * num_ctas;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;

    config.blockDimX = 32 * num_warps;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared_memory;
    config.hStream = stream;
    config.attrs = launchAttr;
    int num_attrs = 0;

    if (launch_pdl != 0) {{
      CUlaunchAttribute pdlAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, .value = 1}};
      launchAttr[num_attrs] = pdlAttr;
      ++num_attrs;
    }}

    if (launch_cooperative_grid != 0) {{
      CUlaunchAttribute coopAttr = {{ .id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE, .value = 1}};
      launchAttr[num_attrs] = coopAttr;
      ++num_attrs;
    }}

    if (num_ctas != 1) {{
      CUlaunchAttribute clusterAttr = {{}};
      clusterAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      clusterAttr.value.clusterDim.x = num_ctas;
      clusterAttr.value.clusterDim.y = 1;
      clusterAttr.value.clusterDim.z = 1;
      launchAttr[num_attrs] = clusterAttr;
      ++num_attrs;

      CUlaunchAttribute clusterSchedulingAttr = {{}};
      clusterSchedulingAttr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      clusterSchedulingAttr.value.clusterSchedulingPolicyPreference = CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
      launchAttr[num_attrs] = clusterSchedulingAttr;
      ++num_attrs;
    }}

    // num_ctas == 16 is non-portable. Does work for H100 and B200 tho
    config.numAttrs = num_attrs;
    if (num_ctas == 16) {{
      CUDA_CHECK(cuFuncSetAttribute(
          function,
          CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
          1
      ));
    }}

    CUDA_CHECK(cuLaunchKernelExHandle(&config, function, params, 0));
  }}
}}

typedef struct _DevicePtrInfo {{
    CUdeviceptr dev_ptr;
    bool valid;
}} DevicePtrInfo;

static PyObject* data_ptr_str = NULL;
static PyObject* py_tensor_map_type = NULL;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ret = PyObject_CallMethodNoArgs(obj, data_ptr_str);
  if (!ret) {{
    PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
    ptr_info.valid = false;
    goto cleanup;
  }}
  if (!PyLong_Check(ret)) {{
    PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
    ptr_info.valid = false;
    goto cleanup;
  }}
  ptr_info.dev_ptr = PyLong_AsUnsignedLongLong(ret);
  if(!ptr_info.dev_ptr)
    return ptr_info;
  uint64_t dev_ptr;
  int status = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, ptr_info.dev_ptr);
  if (status == CUDA_ERROR_INVALID_VALUE) {{
      PyErr_Format(PyExc_ValueError,
                   "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
      ptr_info.valid = false;
  }} else if (status != CUDA_SUCCESS) {{
      CUDA_CHECK(status);  // Catch any other cuda API errors
      ptr_info.valid = false;
  }}
  ptr_info.dev_ptr = dev_ptr;
cleanup:
  Py_XDECREF(ret);
  return ptr_info;

}}

static inline CUtensorMap* getTmaDesc(PyObject *obj) {{
  if (sizeof(CUtensorMap*) != 8) {{
    PyErr_SetString(PyExc_SystemError, "getTmaDesc() requires 64-bit compilation");
    return NULL;
  }}

if (Py_TYPE(obj) != (PyTypeObject*)py_tensor_map_type) {{
    PyErr_Format(PyExc_TypeError, "object must be of type PyCUtensorMap, got %s", Py_TYPE(obj)->tp_name);
    return NULL;
}}

  CUtensorMap* map = &((PyCUtensorMapObject*)obj)->tensorMap;
  uintptr_t align_128 = (uintptr_t)map & (128 - 1);
  if (align_128 != 0) {{
    PyErr_Format(PyExc_ValueError, "CUtensorMap must be aligned to 128B, but got (&map) mod 128 = %ld", align_128);
    return NULL;
  }}
  return map;
}}

static void ensureCudaContext() {{
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {{
    // Ensure device context.
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }}
}}

static uint16_t pack_fp16(double f) {{
    uint16_t result;
    // from https://github.com/python/pythoncapi-compat
#if 0x030600B1 <= PY_VERSION_HEX && PY_VERSION_HEX <= 0x030B00A1 && !defined(PYPY_VERSION)
    _PyFloat_Pack2(f, (unsigned char*)&result, 1);
#else
    PyFloat_Pack2(f, (unsigned char*)&result, 1);
#endif
    return result;
}}

static uint16_t pack_bf16(double f) {{
    float f32 = (float)f;
    uint32_t u32 = *(uint32_t*)&f32;
    return (uint16_t)(u32 >> 16);
}}

static uint32_t pack_fp32(double f) {{
    float f32 = (float)f;
    return *(uint32_t*)&f32;
}}

static uint64_t pack_fp64(double f) {{
    return *(uint64_t*)&f;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  // ensure cuda context is valid before calling any CUDA APIs, e.g. before getPointer calls cuPointerGetAttributes
  ensureCudaContext();

  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  int launch_cooperative_grid;
  int launch_pdl;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *global_scratch_obj = NULL;
  PyObject *profile_scratch_obj = NULL;
  {newline.join([f"{_extracted_type(ty)} _arg{i};" for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ,
                                           &_stream, &_function, &launch_cooperative_grid, &launch_pdl, &global_scratch_obj, &profile_scratch_obj,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook{args_list})) {{
    return NULL;
  }}

  int num_warps, num_ctas, shared_memory;
  if (!PyArg_ParseTuple(kernel_metadata, \"iii\", &num_warps, &num_ctas, &shared_memory)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_enter_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  CUdeviceptr global_scratch = 0;
  if (global_scratch_obj != Py_None) {{
    DevicePtrInfo global_scratch_info = getPointer(global_scratch_obj, -1);
    if (!global_scratch_info.valid) {{
      return NULL;
    }}
    global_scratch = global_scratch_info.dev_ptr;
  }}

  CUdeviceptr profile_scratch = 0;
  if (profile_scratch_obj != Py_None) {{
    DevicePtrInfo profile_scratch_info = getPointer(profile_scratch_obj, -1);
    if (!profile_scratch_info.valid) {{
      return NULL;
    }}
    profile_scratch = profile_scratch_info.dev_ptr;
  }}

  // raise exception asap
  {newline.join(ptr_decls)}
  {newline.join(tma_decls)}
  {newline.join(float_storage_decls)}
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, num_warps, num_ctas, launch_cooperative_grid, launch_pdl, shared_memory, (CUstream)_stream, (CUfunction)_function, global_scratch, profile_scratch{', ' + ', '.join(internal_args_list) if len(internal_args_list) > 0 else ''});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* ret = PyObject_CallOneArg(launch_exit_hook, launch_metadata);
    if (!ret)
      return NULL;
    Py_DECREF(ret);
  }}

  Py_RETURN_NONE;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};

PyMODINIT_FUNC PyInit___triton_launcher(void) {{
  data_ptr_str = PyUnicode_InternFromString("data_ptr");
  if(data_ptr_str == NULL) {{
    return NULL;
  }}
  PyObject* driver_mod = PyImport_ImportModule("triton.backends.nvidia.driver");
  if (driver_mod == NULL) {{
    return NULL;
  }}
  py_tensor_map_type = PyObject_GetAttrString(driver_mod, "PyCUtensorMap");
  if (py_tensor_map_type == NULL) {{
    return NULL;
  }}

  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}
"""
    return src


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def make_tensordesc_arg(arg, metadata):
    if metadata is None:
        # Currently the host side tensor descriptors get decomposed in
        # the frontend to tensor desc, shape, and strides. We have no
        # way to use these shape and strides when processing tensor
        # descriptors which is why we provide our own decomposition
        # above. Sadly this means we have to pass the shape and strides
        # twice.
        return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]

    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1
    padding = 1 if arg.padding == "nan" else 0

    if fp4_padded:
        shape = list(shape)
        shape[-1] *= 2

    cu_tensor_map = triton.runtime.driver.active.utils.fill_tma_descriptor(
        arg.base.data_ptr(),
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
        padding,
    )

    return [cu_tensor_map, *shape, *strides]


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    has_tensor_desc_arg = any(isinstance(sig, str) and sig.startswith("tensordesc") for sig in signature.values())
    if not has_tensor_desc_arg:
        return launcher

    tensordesc_indices = set(
        [i for i, sig in enumerate(signature.values()) if isinstance(sig, str) and sig.startswith("tensordesc")])
    assert not tensordesc_meta or len(tensordesc_meta) == len(tensordesc_indices)
    if not tensordesc_meta:
        tensordesc_meta = [None] * len(tensordesc_indices)

    def inner(*args):
        final_args = list(args[:_BASE_ARGS_FORMAT_LEN])
        tensordesc_idx = 0
        for i, arg in enumerate(args[_BASE_ARGS_FORMAT_LEN:]):
            if i in tensordesc_indices:
                final_args.extend(make_tensordesc_arg(arg, tensordesc_meta[tensordesc_idx]))
                tensordesc_idx += 1
            else:
                final_args.append(arg)
        return launcher(*final_args)

    return inner


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        src = make_launcher(constants, signature, tensordesc_meta)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

        self.num_ctas = getattr(metadata, "num_ctas", 1)
        self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)
        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    global_scratch, profile_scratch, *args)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is None)
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()




这整个文件是 Triton 的 NVIDIA CUDA 后端的关键部分。它的核心任务是创建一个动态的、专门用于启动（Launch）特定 Triton 内核的 C 语言“胶水层”，然后实时编译这个胶水层，并提供一个 Python 对象来调用它。
第 1-13 行: 全局设置和导入
code
Python
import functools, os, subprocess, triton, re
from pathlib import Path
from triton import knobs
from triton.runtime.build import compile_module_from_src
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))
include_dirs = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")
libraries = ['cuda']
PyCUtensorMap = None
导入: 导入了各种必要的模块，包括 triton 自身的模块（如 knobs 配置、compile_module_from_src 编译器、_allocation 内存分配器）和标准库。
路径设置: 定义了一些关键目录路径，例如 include_dirs（包含自定义 C 头文件）和 libdevice_dir（可能包含特定于设备的库）。
全局变量: libraries = ['cuda'] 指定了编译 C 扩展时需要链接 libcuda.so 库。PyCUtensorMap 是一个稍后会从 C 模块加载的类型。
第 16-41 行: libcuda_dirs 函数
这个函数的作用是在系统中找到 libcuda.so.1 所在的目录。这对链接 C 扩展至关重要。
code
Python
@functools.lru_cache()
def libcuda_dirs():
    # ... (代码)
@functools.lru_cache(): 这是一个缓存装饰器。由于 CUDA 驱动库的位置在程序运行期间不会改变，这个函数只需成功执行一次，之后的结果就会被缓存，避免了重复的、昂贵的查找操作。
查找策略:
首先检查 triton.knobs.nvidia.libcuda_path，允许用户手动指定路径。
如果用户未指定，则执行 Linux 命令 /sbin/ldconfig -p。这个命令会列出系统链接器缓存中所有已知的共享库及其位置。
代码解析 ldconfig 的输出，找到包含 libcuda.so.1 的行，并提取其目录。
如果 ldconfig 没找到，它会尝试检查 LD_LIBRARY_PATH 环境变量。
如果最终还是找不到，它会 assert 失败，并打印一条非常详细和有用的错误消息，指导用户如何解决问题。
第 44-47 行: library_dirs 函数
code
Python
@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]
这个函数简单地将本地的 libdevice_dir 和通过 libcuda_dirs() 找到的系统目录合并在一起，形成一个完整的库搜索路径列表。同样，它也被缓存了。
第 52-73 行: CudaUtils 类
这是一个单例 (Singleton) 工具类，用于加载一个预先编写好的 C 模块 (driver.c)，该模块提供了一些 Python 无法直接访问的底层 CUDA API 功能。
code
Python
class CudaUtils(object):
    def __new__(cls): # ... 实现单例模式
    def __init__(self):
        mod = compile_module_from_src(...) # 编译 driver.c
        # ...
        self.load_binary = mod.load_binary
        # ... (加载其他函数)
__new__: 实现了单例模式，确保整个程序中只有一个 CudaUtils 实例。
__init__:
读取 driver.c 文件的源代码。
调用我们之前分析过的 compile_module_from_src 函数，将这个 C 源代码编译成一个名为 cuda_utils 的 Python 扩展模块。
从编译好的 mod 模块中，提取出如 load_binary、get_device_properties 等 C 函数的 Python 包装器，并将它们作为 CudaUtils 实例的属性。
第 80-359 行: make_launcher 函数
这是整个文件的核心。它是一个代码生成器。它的输入是 Triton 内核的元数据（常量、参数签名），输出是一个巨大的 C 语言源代码字符串。这个 C 代码专门用于启动这个特定的内核。
ty_to_cpp 函数 (第 80-101 行): 将 Triton 的类型字符串 (如 '*fp32', 'i32') 映射到 C/CUDA 对应的类型 (如 'CUdeviceptr', 'int32_t')。
FLOAT_STORAGE_TYPE 和 FLOAT_PACK_FUNCTION 字典 (第 104-115 行): 定义了半精度/单精度浮点数在 C 中的存储类型（如 uint16_t）以及用于将 Python float (即 C double) 打包成这些格式的函数名。
签名处理 (第 121-188 行):
_expand_signature: 处理复杂的参数类型，特别是 tensordesc（张量描述符），它会被展开成多个基础类型的参数（如指针、形状、步长）。
_flatten_signature: 处理元组类型的参数，将它们“拍平”成一个扁平的参数列表。
format_of: 根据参数类型生成一个格式化字符串，供 Python C API 的 PyArg_ParseTuple 函数使用，用于从 Python 元组中解析参数。
C 代码生成 (第 215 行 onwards): 使用 Python 的 f-string 功能，将前面处理好的参数声明、类型转换、函数调用等部分动态地嵌入到一个巨大的 C 代码模板中。
我们来分析一下生成的 C 代码的关键部分：
gpuAssert 和 CUDA_CHECK: 用于 CUDA API 调用的错误检查宏，如果 API 调用失败，它会设置 Python 异常。
getLaunchKernelExHandle: 使用 dlopen 和 dlsym 在运行时动态加载 libcuda.so.1 并查找 cuLaunchKernelEx 函数的地址。这样做的好处是避免了在编译时硬链接 CUDA 库，增加了可移植性。
_launch: 这是一个 C 函数，它接收网格维度、线程块维度、共享内存大小、CUDA 流、函数句柄以及所有内核参数，然后配置 CUlaunchConfig 结构体，并最终调用 cuLaunchKernelEx 来启动内核。
getPointer: 一个非常重要的辅助函数。它负责将一个 Python 对象转换为 CUDA 设备指针 (CUdeviceptr)。它能处理两种情况：一个表示地址的 Python 整数，或者一个拥有 .data_ptr() 方法的对象（比如 PyTorch Tensor）。它还使用 cuPointerGetAttribute 来验证这个指针确实是 GPU 上的有效指针。
launch: 这是最终暴露给 Python 的 C 函数。它：
使用 PyArg_ParseTuple 和之前生成的格式化字符串来解析从 Python 传入的参数。
调用 getPointer 等函数来转换参数。
使用 Py_BEGIN_ALLOW_THREADS 释放 Python 的全局解释器锁 (GIL)，因为 CUDA 内核启动是一个耗时且阻塞的操作，释放 GIL 可以让其他 Python 线程运行。
调用 C 函数 _launch 来实际启动内核。
使用 Py_END_ALLOW_THREADS 重新获取 GIL。
模块定义: 文件末尾是标准的 Python C 扩展模块定义样板代码 (PyMethodDef, PyModuleDef, PyInit___triton_launcher)。
第 363-412 行: 张量描述符处理
make_tensordesc_arg 和 wrap_handle_tensordesc 是专门用来处理高级 CUDA 特性——张量描述符（Tensor Memory Accessor, TMA）的。
make_tensordesc_arg: 将一个高层的 Triton 张量描述符对象分解成 C API 所需的底层信息（基地址、形状、步长、填充等）。
wrap_handle_tensordesc: 这是一个装饰器（高阶函数）。它包装了编译好的 launch 函数。如果内核签名中包含张量描述符，这个包装器会拦截调用，将张量描述符参数用 make_tensordesc_arg 进行转换，然后再调用真正的 launch 函数。
第 415-455 行: CudaLauncher 类
这个类是最终暴露给 Triton 运行时的接口。
__init__:
接收内核的元数据 (src, metadata)。
调用 make_launcher 来生成专门用于此内核的 C 源代码。
调用 compile_module_from_src 来实时编译这份 C 代码，得到一个 Python 模块。
从模块中获取 launch 函数，并使用 wrap_handle_tensordesc 对其进行包装。
存储一些元数据，如 num_ctas（每个线程块簇的线程块数）、临时内存（scratch memory）大小等。
__call__:
这是启动内核的实际入口点。
它负责根据元数据分配所需的临时 GPU 内存（global_scratch 和 profile_scratch）。
最后，它调用 self.launch（也就是那个编译好的、被包装过的 C 函数），传入网格维度、流、函数句柄以及所有用户参数，从而启动 GPU 内核。
第 458-500 行: CudaDriver 类
这是一个驱动类，它将上述所有功能集成起来，为 Triton 的顶层 API 提供了一个统一的后端接口。
它初始化 CudaUtils。
它指定 CudaLauncher 作为其启动器类。
它提供了查询当前设备信息（get_current_target）、与 PyTorch 集成（get_active_torch_device）、提供基准测试工具（get_benchmarker）等方法。
总结
这个文件是一个高度复杂的即时代码生成和编译系统，其工作流程如下：
分析内核: CudaLauncher 接收一个 Triton 内核的抽象描述（签名、元数据）。
生成 C 代码: make_launcher 根据这个描述，动态生成一个 C 语言源文件。这个 C 文件包含一个名为 launch 的 Python C API 函数，该函数被硬编码为知道如何接收这个特定内核的所有参数，并将它们传递给 CUDA 驱动 API。
实时编译: compile_module_from_src 调用系统 C 编译器（如 GCC/Clang）将生成的 C 代码编译成一个共享库（.so 文件），并将其作为 Python 模块加载到内存中。
封装和调用: CudaLauncher 实例持有了这个编译好的 launch 函数的句柄。当用户调用 launcher(...) 时，实际上是在调用这个刚刚在运行时编译出来的 C 函数，从而以最高效的方式启动 GPU 计算任务。
