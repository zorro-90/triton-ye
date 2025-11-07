from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sysconfig
import tempfile

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str]) -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    cc = os.environ.get("CC")
    if cc is None:
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError(
                "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None) -> ModuleType:
    key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")

    if cache_path is not None:
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    return _load_module_from_path(name, cache_path)





这个文件的核心功能是在运行时从一个 C 语言源代码字符串动态编译、缓存并加载一个 Python 扩展模块。这是一个典型的即时编译（JIT）工具链中的一环，常见于需要高性能计算的库（如 Triton）。
第 3-13 行: 导入
code
Python
import functools
import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sysconfig
import tempfile
from types import ModuleType
from .cache import get_cache_manager
from .. import knobs
标准库导入：
functools: 用于 lru_cache，一个内存缓存装饰器。
hashlib: 用于计算 SHA256 哈希值，这是缓存系统的关键。
importlib.util: 现代的、推荐的用于动态加载模块的方式。
logging: 用于记录警告和错误。
os, shutil, subprocess, sysconfig, tempfile: 都是与操作系统交互的模块，用于文件路径操作、查找程序（如 gcc）、执行外部命令（编译）、获取 Python 配置和管理临时文件。
types.ModuleType: 用于类型提示，明确指出函数返回的是一个模块对象。
相对导入：
from .cache import get_cache_manager: 从同级目录的 cache.py 文件中导入缓存管理函数。
from .. import knobs: 从父级目录的 __init__.py 中导入 knobs 对象。knobs 看起来是一个全局配置对象，允许用户自定义某些行为（比如自定义编译实现）。
第 16-47 行: _build 函数
这是一个内部辅助函数，负责执行实际的编译命令。
code
Python
def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str]) -> str:
接收模块名、源文件路径、源文件目录以及各种编译器标志作为参数，返回编译后的共享库（.so 文件）的路径。
code
Python
if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
这是一个扩展点。它检查 knobs.build.impl 是否被用户设置。如果设置了，就调用用户自定义的编译函数，而不是执行下面的默认逻辑。这使得库可以灵活地适应不同的编译环境。使用了 := (walrus operator) 来简化代码。
code
Python
suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, f'{name}{suffix}')
获取当前平台标准的扩展模块后缀（如 Linux 上的 .so，Windows 上的 .pyd），并构造出最终的输出文件路径。
code
Python
cc = os.environ.get("CC")
    if cc is None:
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError(...)
查找 C 编译器：
首先检查 CC 环境变量，这是配置编译器的标准方式。
如果 CC 未设置，使用 shutil.which 在系统 PATH 中查找 clang 和 gcc。
优先使用 gcc。
如果两者都找不到，就抛出一个清晰的错误。
code
Python
py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    ...
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
获取 Python C 头文件（如 Python.h）所在的目录，并将其添加到包含目录列表中。这是编译 Python C 扩展所必需的。
code
Python
cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    cc_cmd += [f'-l{lib}' for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
构建编译器命令：
[cc, src, "-o", so]: 基本结构 compiler input_file -o output_file。
-O3: 高级别优化。
-shared: 生成一个共享库。
-fPIC: 生成位置无关代码，这是共享库所必需的。
-Wno-psabi: 忽略一个特定的 GCC ABI 警告，可能是为了兼容性。
-l, -L, -I: 分别用于链接库、指定库搜索目录和指定头文件搜索目录。
code
Python
subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so
使用 subprocess.check_call 执行编译命令。如果编译失败（编译器返回非零退出码），这个函数会抛出异常。stdout=subprocess.DEVNULL 会丢弃编译器的标准输出，保持终端清洁。
成功后，返回编译产物 .so 文件的路径。
第 50-54 行: platform_key 函数
code
Python
@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])
@functools.lru_cache: 装饰器，将函数的结果缓存起来。因为程序的运行平台不会改变，所以这个函数只需要计算一次。
作用：生成一个能唯一标识当前操作系统和CPU架构的字符串（例如 "x86_64,Linux,64bit,ELF"）。这个 key 至关重要，因为它能确保为不同平台编译的二进制文件被缓存在不同的地方。
第 57-64 行: _load_module_from_path 函数
code
Python
def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
这是一个标准的、现代的 Python 方式，用于从任意路径加载一个动态链接库作为 Python 模块。
它首先创建一个模块“规范”（spec），然后根据规范创建一个空的模块对象（mod），最后由加载器（spec.loader）执行模块代码并填充该对象。
第 67-95 行: compile_module_from_src 函数
这是对外暴露的主函数，它将所有部分串联起来。
code
Python
def compile_module_from_src(...):
接收 C 源码字符串、模块名和各种可选的编译器标志。
code
Python
key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
核心缓存逻辑：
将源代码和平台标识拼接在一起。
计算这个组合字符串的 SHA256 哈希值。
这个哈希值 key 是此编译任务的唯一ID。如果源代码或平台变了，key 就会变。
使用这个 key 获取一个缓存管理器实例。
code
Python
cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is not None:
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            ... # Log a warning
```*   **缓存命中 (Cache Hit)**：
    *   尝试从缓存中获取文件。如果 `get_file` 返回了一个路径，说明之前已经编译过了。
    *   尝试加载这个已缓存的文件。
    *   如果加载失败（例如，缓存文件损坏或与当前 Python 版本不兼容），则记录一个警告，然后继续执行下面的“缓存未命中”逻辑。

```python
    # (If cache miss)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(...)
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)
缓存未命中 (Cache Miss)：
with tempfile.TemporaryDirectory(): 创建一个临时的、安全的目录，它会在代码块结束后自动被删除。
将传入的 src 字符串写入一个临时的 .c 文件中。
调用 _build 函数来编译这个 .c 文件，得到 .so 文件的路径。
读取编译好的 .so 文件的二进制内容，并使用 cache.put 方法将其存入缓存。
code
Python
return _load_module_from_path(name, cache_path)
最后，无论是在缓存命中后，还是在缓存未命中、新编译完成后，都调用 _load_module_from_path 来加载模块并返回给调用者。
总结
这个模块实现了一个健壮、高效且可扩展的 JIT 编译流程：
唯一标识：通过对源码和平台信息进行哈希，为每个编译任务创建唯一 ID。
缓存：在编译前检查缓存，如果存在有效的编译产物则直接加载，大大加快了重复执行的速度。
编译：如果缓存未命中，则安全地在临时目录中执行编译。
加载：使用标准库动态加载编译好的模块。
健壮性：能处理缓存文件损坏的情况，并能自动查找编译器。
可扩展性：通过 knobs 对象允许用户提供自己的编译实现。
