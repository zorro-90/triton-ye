from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future

    def result(self):
        if self.kernel is not None:
            return self.kernel

        kernel = self.future.result()
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel


class AsyncCompileMode:

    def __init__(self, executor: Executor):
        self.executor = executor
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Finalize any outstanding compiles
        for future in as_completed(self.raw_futures):
            self.future_kernels[future._key].result()
        active_mode.set(None)





这是一个设计精良的框架，用于实现异步编译。在一些需要即时编译（JIT）代码（例如 Triton、Numba 或机器学习框架中的计算图）的场景中，编译本身可能是一个耗时的操作。这个框架允许程序提交一个编译任务后，不阻塞主线程，可以继续执行其他任务，然后在需要编译结果时再等待它完成。
第 1-4 行: 导入
code
Python
from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar
from __future__ import annotations: 这是一个未来特性导入，它允许在类型提示中使用尚未完全定义的类名。例如，在 AsyncCompileMode 类的方法中，可以把返回类型提示为 AsyncCompileMode，而不需要用字符串 "AsyncCompileMode"。
from typing import Callable, Optional:
Callable: 类型提示，表示一个可调用对象，通常是一个函数。
Optional: 类型提示，表示一个值可以是指定类型或 None。
from concurrent.futures import Executor, as_completed, Future: 这是实现并发的核心模块。
Executor: 一个抽象基类，提供了异步执行调用的方法。常见的实现有 ThreadPoolExecutor (使用线程) 和 ProcessPoolExecutor (使用进程)。
Future: 代表一个异步计算的最终结果。你可以提交一个任务给 Executor，它会立即返回一个 Future 对象。你可以稍后查询这个 Future 来获取结果或查看它是否已完成。
as_completed: 一个函数，接收一个 Future 对象的迭代器，然后返回一个新的迭代器，该迭代器会在任何一个 Future 完成时立即产生 (yield) 它。这对于处理一组异步任务的结果非常有用，因为你不需要按提交的顺序等待它们。
from contextvars import ContextVar: 导入上下文变量，用于在特定执行上下文中安全地存储状态，避免使用容易出错的全局变量。
第 6 行: 上下文变量
code
Python
active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)
这里定义了一个名为 active_mode 的上下文变量。
它的作用是跟踪当前是否有一个 AsyncCompileMode 处于活动状态。
默认值为 None，表示默认情况下没有活动的异步编译模式。
使用 ContextVar 可以确保这个状态是与当前上下文（如当前线程或异步任务）绑定的，避免了多线程环境下的状态污染。
第 9-22 行: FutureKernel 类
这个类是一个“智能包装器”，它包装了 concurrent.futures.Future 对象，并增加了一些额外的逻辑。
code
Python
class FutureKernel:

    def __init__(self, finalize_compile: Callable, future: Future):
        self.finalize_compile = finalize_compile
        self.kernel = None
        self.future = future
__init__: 构造函数。
finalize_compile: 接收一个函数。这个函数将在编译完成后、返回最终结果之前被调用。这可以用于缓存编译结果、注册内核等后处理步骤。
future: 从 Executor 返回的原始 Future 对象。
self.kernel: 用于缓存编译完成的内核，初始为 None。
code
Python
def result(self):
        if self.kernel is not None:
            return self.kernel

        kernel = self.future.result()
        self.finalize_compile(kernel)
        self.kernel = kernel
        return kernel
result(): 获取编译结果的方法。
if self.kernel is not None: 这是一个简单的缓存检查。如果之前已经获取过结果并缓存了，就直接返回，避免重复工作。
kernel = self.future.result(): 这是关键的阻塞点。调用原始 Future 对象的 result() 方法会等待，直到后台的编译任务完成，然后返回其结果。
self.finalize_compile(kernel): 一旦拿到编译好的 kernel，就调用初始化时传入的 finalize_compile 函数进行后处理。
self.kernel = kernel: 将最终的 kernel 缓存起来。
return kernel: 返回最终结果。
第 25-60 行: AsyncCompileMode 类
这是整个框架的核心控制器，它被设计成一个上下文管理器（可以使用 with 语句）。
code
Python
class AsyncCompileMode:

    def __init__(self, executor: Executor):
        self.executor = executor
        self.raw_futures = []
        self.future_kernels = {}```
*   **`__init__`**:
    *   `executor`: 接收一个 `Executor` 实例（如 `ThreadPoolExecutor`），编译任务将提交到这里执行。
    *   `self.raw_futures`: 一个列表，用于存储所有被提交的原始 `Future` 对象。这主要用于 `__exit__` 中的 `as_completed`。
    *   `self.future_kernels`: 一个字典，用于缓存和去重。它将一个唯一的 `key` 映射到对应的 `FutureKernel` 对象。

```python
    def submit(self, key, compile_fn, finalize_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(finalize_fn, future)
        self.future_kernels[key] = future_kernel
        return future_kernel
submit(): 提交一个编译任务。
key: 任务的唯一标识符，用于去重。例如，可以是一个编译选项的哈希值。
compile_fn: 要在后台执行的编译函数。
finalize_fn: 编译完成后要调用的后处理函数。
if future is not None: 首先检查 key 是否已存在于缓存中。如果存在，说明同样内容的编译任务已经被提交了，直接返回已有的 FutureKernel，避免重复编译。
future = self.executor.submit(compile_fn): 如果是新任务，就把它提交给 Executor。这是一个非阻塞调用，它会立即返回一个 Future 对象。
future._key = key: 将 key 附加到 Future 对象上。这是一个常见的技巧，用于在稍后处理 Future 时能够识别它。
self.raw_futures.append(future): 将原始 Future 存入列表。
future_kernel = FutureKernel(...): 创建 FutureKernel 包装器。
self.future_kernels[key] = future_kernel: 将新的 FutureKernel 存入缓存字典。
return future_kernel: 立即返回 FutureKernel 对象，主程序可以继续执行。
code
Python
def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self
__enter__: with 语句开始时调用。
if active_mode.get() is not None: 检查上下文变量，如果已经有另一个 AsyncCompileMode 处于活动状态，就抛出异常。这可以防止 with 块的意外嵌套。
active_mode.set(self): 将当前实例设置为活动的模式。
return self: 返回自身，以便在 with 块内通过 as 关键字引用。
code
Python
def __exit__(self, exc_type, exc_value, traceback):
        # Finalize any outstanding compiles
        for future in as_completed(self.raw_futures):
            self.future_kernels[future._key].result()
        active_mode.set(None)
__exit__: with 语句结束时调用，无论是正常结束还是发生异常。
这是自动同步点。
for future in as_completed(self.raw_futures): 遍历所有已提交的、尚未处理的 Future。as_completed 会在每个 Future 完成时立即返回它。
self.future_kernels[future._key].result(): 对于每个完成的 Future，找到它对应的 FutureKernel 并调用 .result()。这确保了编译任务不仅完成，而且其 finalize_fn 也被执行了。
这个循环保证了在退出 with 块时，所有提交的编译任务都已经彻底完成。
active_mode.set(None): 清理上下文变量，表示异步编译模式已结束。
总结与实践
这个系统的工作流程如下：
创建一个线程池 executor = ThreadPoolExecutor()。
使用 with AsyncCompileMode(executor) as mode: 进入异步编译模式。
在 with 块内部，多次调用 mode.submit(key, compile_func, finalize_func)。这些调用会立即返回 FutureKernel 对象，编译在后台线程中开始。
程序可以继续执行其他操作，或者在需要某个特定内核时调用 future_kernel.result() 来阻塞等待它完成。
当 with 块结束时，__exit__ 方法会自动执行，它会等待所有尚未完成的编译任务全部结束，确保没有遗漏。
这种设计模式的优点是：
隐藏了并发的复杂性：用户只需使用简单的 with 语句和 submit 方法。
高效：通过 as_completed 并行处理结果，并通过字典缓存避免了重复工作。
健壮：通过 ContextVar 和 __enter__ 中的检查，避免了状态管理错误。
保证同步：__exit__ 确保了在程序继续往下执行之前，所有相关的后台任务都已完成，避免了悬空任务。
