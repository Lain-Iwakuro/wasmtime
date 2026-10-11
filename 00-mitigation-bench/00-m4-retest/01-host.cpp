// WASI threads host for the Apple mitigation experiment.
// ABI follows Wasmtime v43 crates/wasi-threads: import wasi.thread-spawn(i32)->i32,
// export wasi_thread_start(i32,i32)->(), thread ids in 1..=0x1FFFFFFF, negative means failure.

#include <wasmtime.h>

#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

enum class StopKind { Returned, Exit, Trap, Error };

struct Stop {
  StopKind kind = StopKind::Returned;
  int code = 0;
  std::string message;
};

struct WorkerStop {
  int32_t tid = 0;
  Stop stop;
};

void delete_engine(wasm_engine_t *p) { wasm_engine_delete(p); }
void delete_module(wasmtime_module_t *p) { wasmtime_module_delete(p); }
void delete_linker(wasmtime_linker_t *p) { wasmtime_linker_delete(p); }
void delete_store(wasmtime_store_t *p) { wasmtime_store_delete(p); }
void delete_pre(wasmtime_instance_pre_t *p) { wasmtime_instance_pre_delete(p); }
void delete_memory(wasmtime_sharedmemory_t *p) { wasmtime_sharedmemory_delete(p); }

template <class T, void (*Delete)(T *)>
class Owned {
 public:
  Owned() = default;
  explicit Owned(T *p) : p_(p) {}
  ~Owned() { reset(); }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  Owned(Owned &&other) noexcept : p_(other.p_) { other.p_ = nullptr; }
  Owned &operator=(Owned &&other) noexcept {
    if (this != &other) {
      reset();
      p_ = other.p_;
      other.p_ = nullptr;
    }
    return *this;
  }

  T *get() const { return p_; }
  T *release() {
    T *out = p_;
    p_ = nullptr;
    return out;
  }
  void reset(T *next = nullptr) {
    if (p_ != nullptr) {
      Delete(p_);
    }
    p_ = next;
  }

 private:
  T *p_ = nullptr;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string error_text(wasmtime_error_t *error) {
  if (error == nullptr) {
    return "";
  }
  wasm_name_t message;
  wasmtime_error_message(error, &message);
  std::string out(message.data, message.size);
  wasm_byte_vec_delete(&message);
  wasmtime_error_delete(error);
  return out;
}

std::string trap_text(wasm_trap_t *trap) {
  if (trap == nullptr) {
    return "";
  }
  wasm_message_t message;
  wasm_trap_message(trap, &message);
  std::string out(message.data, message.size);
  wasm_byte_vec_delete(&message);
  wasm_trap_delete(trap);
  return out;
}

void check(wasmtime_error_t *error, const char *context) {
  if (error != nullptr) {
    fail(std::string(context) + ": " + error_text(error));
  }
}

std::string name_of(const wasm_name_t *name) {
  if (name == nullptr || name->data == nullptr || name->size == 0) {
    return "";
  }
  return std::string(name->data, name->size);
}

const char *extern_kind_name(wasm_externkind_t kind) {
  switch (kind) {
    case WASM_EXTERN_FUNC:
      return "func";
    case WASM_EXTERN_GLOBAL:
      return "global";
    case WASM_EXTERN_TABLE:
      return "table";
    case WASM_EXTERN_MEMORY:
      return "memory";
    default:
      return "other";
  }
}

const char *stop_kind_name(StopKind kind) {
  switch (kind) {
    case StopKind::Returned:
      return "return";
    case StopKind::Exit:
      return "exit";
    case StopKind::Trap:
      return "trap";
    case StopKind::Error:
      return "error";
  }
  return "error";
}

Stop classify(wasmtime_error_t *error, wasm_trap_t *trap) {
  Stop stop;
  if (error != nullptr) {
    int status = 0;
    if (wasmtime_error_exit_status(error, &status)) {
      wasmtime_error_delete(error);
      stop.kind = StopKind::Exit;
      stop.code = status;
      return stop;
    }
    stop.kind = StopKind::Error;
    stop.message = error_text(error);
    return stop;
  }
  if (trap != nullptr) {
    stop.kind = StopKind::Trap;
    stop.message = trap_text(trap);
    return stop;
  }
  stop.kind = StopKind::Returned;
  return stop;
}

bool func_is_i32_i32_to_void(const wasm_functype_t *type) {
  const wasm_valtype_vec_t *params = wasm_functype_params(type);
  const wasm_valtype_vec_t *results = wasm_functype_results(type);
  return params->size == 2 && results->size == 0 &&
         wasm_valtype_kind(params->data[0]) == WASM_I32 &&
         wasm_valtype_kind(params->data[1]) == WASM_I32;
}

struct Options {
  std::string mode;
  std::string setting;
  std::string v4_policy = "legacy";
  std::string wasm_path;
  std::vector<std::string> guest_argv;
  int timeout_sec = 60;
  int max_threads = 64;
  bool has_preopen = false;
  std::string preopen_host;
  std::string preopen_guest;
  // When set, print CLOCK_MONOTONIC nanoseconds from the _start call through
  // worker join. Engine init, wasm read, JIT, main instance, and export
  // resolution stay outside that interval.
  bool time_exec = false;
};

void usage() {
  fprintf(stderr,
          "usage: host --mode none|cas_csdb|cas-csdb|ssbb --wasm FILE "
          "[--v4-policy legacy|frame|ranges|ranges_cfg] [--timeout SEC] [--max-threads N] [--dir HOST[::GUEST]] [--time] "
          "[-- GUEST_ARGS]\n");
}

std::string setting_for_mode(const std::string &mode) {
  if (mode == "none") {
    return "none";
  }
  if (mode == "cas_csdb" || mode == "cas-csdb") {
    return "cas_csdb";
  }
  if (mode == "ssbb") {
    return "ssbb";
  }
  fail("unknown mitigation mode: " + mode);
}

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto need = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        fail(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (arg == "--") {
      for (int j = i + 1; j < argc; j++) {
        opt.guest_argv.emplace_back(argv[j]);
      }
      break;
    } else if (arg == "--mode") {
      opt.mode = need("--mode");
    } else if (arg == "--v4-policy") {
      opt.v4_policy = need("--v4-policy");
    } else if (arg == "--wasm") {
      opt.wasm_path = need("--wasm");
    } else if (arg == "--timeout") {
      opt.timeout_sec = std::stoi(need("--timeout"));
    } else if (arg == "--max-threads") {
      opt.max_threads = std::stoi(need("--max-threads"));
    } else if (arg == "--time") {
      opt.time_exec = true;
    } else if (arg == "--dir") {
      std::string spec = need("--dir");
      auto split = spec.find("::");
      opt.has_preopen = true;
      if (split == std::string::npos) {
        opt.preopen_host = spec;
        opt.preopen_guest = "/";
      } else {
        opt.preopen_host = spec.substr(0, split);
        opt.preopen_guest = spec.substr(split + 2);
      }
    } else if (arg == "--help") {
      usage();
      exit(0);
    } else {
      fail("unknown argument: " + arg);
    }
  }
  if (opt.mode.empty() || opt.wasm_path.empty()) {
    usage();
    fail(" --mode and --wasm are required");
  }
  if (opt.timeout_sec <= 0) {
    fail("--timeout must be positive");
  }
  if (opt.max_threads < 0) {
    fail("--max-threads must be >= 0");
  }
  opt.setting = setting_for_mode(opt.mode);
  if (opt.v4_policy != "legacy" && opt.v4_policy != "frame" &&
      opt.v4_policy != "ranges" && opt.v4_policy != "ranges_cfg") {
    fail("unknown v4 policy: " + opt.v4_policy);
  }
  if (opt.v4_policy != "legacy" && opt.setting != "ssbb") {
    fail("non-legacy --v4-policy requires --mode ssbb");
  }
  opt.guest_argv.insert(opt.guest_argv.begin(), opt.wasm_path);
  return opt;
}

class Watchdog {
 public:
  explicit Watchdog(int timeout_sec) {
    worker_ = std::thread([this, timeout_sec] {
      std::unique_lock<std::mutex> lock(mu_);
      if (!cv_.wait_for(lock, std::chrono::seconds(timeout_sec),
                        [this] { return done_; })) {
        fprintf(stderr, "timeout: guest run exceeded %d s\n", timeout_sec);
        fflush(stderr);
        _exit(124);
      }
    });
  }

  ~Watchdog() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  Watchdog(const Watchdog &) = delete;
  Watchdog &operator=(const Watchdog &) = delete;

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread worker_;
};

struct Runtime {
  const Options *opt = nullptr;
  wasm_engine_t *engine = nullptr;
  wasmtime_module_t *module = nullptr;
  wasmtime_instance_pre_t *pre = nullptr;

  std::mutex mu;
  std::condition_variable cv;
  int in_flight = 0;
  int starting = 0;
  std::atomic<int32_t> next_tid{0};
  std::vector<std::thread> workers;
  std::vector<WorkerStop> outcomes;
  int spawn_count = 0;
  bool entry_ok = false;

  bool thread_entry_ok() const {
    wasm_exporttype_vec_t exports;
    wasmtime_module_exports(module, &exports);
    bool ok = false;
    for (size_t i = 0; i < exports.size; i++) {
      if (name_of(wasm_exporttype_name(exports.data[i])) != "wasi_thread_start") {
        continue;
      }
      const wasm_externtype_t *ty = wasm_exporttype_type(exports.data[i]);
      if (wasm_externtype_kind(ty) != WASM_EXTERN_FUNC) {
        break;
      }
      ok = func_is_i32_i32_to_void(wasm_externtype_as_functype_const(ty));
      break;
    }
    wasm_exporttype_vec_delete(&exports);
    return ok;
  }

  int32_t allocate_tid() {
    int32_t cur = next_tid.load(std::memory_order_relaxed);
    while (cur <= 0x1ffffffe) {
      if (next_tid.compare_exchange_weak(cur, cur + 1, std::memory_order_relaxed)) {
        return cur + 1;
      }
    }
    return -1;
  }

  void apply_wasi(wasmtime_context_t *context) const {
    wasi_config_t *wasi = wasi_config_new();
    if (wasi == nullptr) {
      fail("wasi_config_new failed");
    }
    std::vector<const char *> argv;
    argv.reserve(opt->guest_argv.size());
    for (const auto &arg : opt->guest_argv) {
      argv.push_back(arg.c_str());
    }
    if (!wasi_config_set_argv(wasi, argv.size(), argv.data())) {
      wasi_config_delete(wasi);
      fail("wasi_config_set_argv failed");
    }
    wasi_config_inherit_stdin(wasi);
    wasi_config_inherit_stdout(wasi);
    wasi_config_inherit_stderr(wasi);
    if (opt->has_preopen) {
      wasi_dir_perms dir_perms =
          WASMTIME_WASI_DIR_PERMS_READ | WASMTIME_WASI_DIR_PERMS_WRITE;
      wasi_file_perms file_perms =
          WASMTIME_WASI_FILE_PERMS_READ | WASMTIME_WASI_FILE_PERMS_WRITE;
      if (!wasi_config_preopen_dir(wasi, opt->preopen_host.c_str(),
                                  opt->preopen_guest.c_str(), dir_perms,
                                  file_perms)) {
        wasi_config_delete(wasi);
        fail("wasi_config_preopen_dir failed for " + opt->preopen_host);
      }
    }
    // set_wasi takes ownership even when it returns an error.
    wasmtime_error_t *error = wasmtime_context_set_wasi(context, wasi);
    check(error, "wasmtime_context_set_wasi");
  }

  Stop call_export(wasmtime_context_t *context, const wasmtime_instance_t *instance,
                   const char *name, const wasmtime_val_t *args, size_t nargs) {
    wasmtime_extern_t item;
    if (!wasmtime_instance_export_get(context, instance, name, strlen(name), &item)) {
      Stop stop;
      stop.kind = StopKind::Error;
      stop.message = std::string("missing export ") + name;
      return stop;
    }
    if (item.kind != WASMTIME_EXTERN_FUNC) {
      wasmtime_extern_delete(&item);
      Stop stop;
      stop.kind = StopKind::Error;
      stop.message = std::string("export is not a func: ") + name;
      return stop;
    }
    wasm_trap_t *trap = nullptr;
    wasmtime_error_t *error = wasmtime_func_call(context, &item.of.func, args, nargs,
                                                 nullptr, 0, &trap);
    wasmtime_extern_delete(&item);
    return classify(error, trap);
  }

  void run_worker(int32_t tid, int32_t start_arg) {
    WorkerStop outcome;
    outcome.tid = tid;
    try {
      Owned<wasmtime_store_t, delete_store> store(wasmtime_store_new(engine, nullptr, nullptr));
      if (store.get() == nullptr) {
        fail("wasmtime_store_new failed");
      }
      wasmtime_context_t *context = wasmtime_store_context(store.get());
      apply_wasi(context);
      wasmtime_instance_t instance;
      wasm_trap_t *trap = nullptr;
      wasmtime_error_t *error =
          wasmtime_instance_pre_instantiate(pre, context, &instance, &trap);
      if (error != nullptr || trap != nullptr) {
        outcome.stop = classify(error, trap);
        outcome.stop.message = "worker instantiate: " + outcome.stop.message;
      } else {
        wasmtime_val_t args[2];
        args[0].kind = WASMTIME_I32;
        args[0].of.i32 = tid;
        args[1].kind = WASMTIME_I32;
        args[1].of.i32 = start_arg;
        outcome.stop = call_export(context, &instance, "wasi_thread_start", args, 2);
      }
    } catch (const std::exception &ex) {
      outcome.stop.kind = StopKind::Error;
      outcome.stop.message = ex.what();
    } catch (...) {
      outcome.stop.kind = StopKind::Error;
      outcome.stop.message = "unknown worker exception";
    }

    {
      std::lock_guard<std::mutex> lock(mu);
      outcomes.push_back(std::move(outcome));
      in_flight--;
    }
  }

  int32_t spawn(int32_t start_arg) {
    if (pre == nullptr) {
      fprintf(stderr, "thread-spawn before instance_pre was ready\n");
      return -1;
    }
    if (!entry_ok) {
      fprintf(stderr, "thread-spawn: missing wasi_thread_start(i32,i32)->()\n");
      return -1;
    }

    int32_t tid = -1;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (in_flight >= opt->max_threads) {
        fprintf(stderr, "thread-spawn refused: in_flight=%d max=%d\n", in_flight,
                opt->max_threads);
        return -1;
      }
      tid = allocate_tid();
      if (tid < 0) {
        fprintf(stderr, "thread-spawn refused: thread id space exhausted\n");
        return -1;
      }
      in_flight++;
      starting++;
    }

    try {
      std::thread worker([this, tid, start_arg] { run_worker(tid, start_arg); });
      {
        std::lock_guard<std::mutex> lock(mu);
        workers.push_back(std::move(worker));
        starting--;
        spawn_count++;
      }
      cv.notify_all();
      return tid;
    } catch (const std::exception &ex) {
      fprintf(stderr, "thread-spawn host thread failed: %s\n", ex.what());
    } catch (...) {
      fprintf(stderr, "thread-spawn host thread failed\n");
    }

    {
      std::lock_guard<std::mutex> lock(mu);
      starting--;
      in_flight--;
    }
    cv.notify_all();
    return -1;
  }

  void join_workers() {
    for (;;) {
      std::thread worker;
      {
        std::unique_lock<std::mutex> lock(mu);
        while (workers.empty() && starting > 0) {
          cv.wait(lock);
        }
        if (workers.empty()) {
          break;
        }
        worker = std::move(workers.back());
        workers.pop_back();
      }
      if (worker.joinable()) {
        worker.join();
      }
    }
  }
};

extern "C" wasm_trap_t *on_thread_spawn(void *env, wasmtime_caller_t *,
                                        const wasmtime_val_t *args, size_t nargs,
                                        wasmtime_val_t *results, size_t nresults) {
  auto *runtime = static_cast<Runtime *>(env);
  int32_t tid = -1;
  try {
    if (nargs == 1 && nresults == 1 && args[0].kind == WASMTIME_I32) {
      tid = runtime->spawn(args[0].of.i32);
    } else {
      fprintf(stderr, "thread-spawn called with an unexpected signature\n");
    }
  } catch (const std::exception &ex) {
    fprintf(stderr, "thread-spawn exception: %s\n", ex.what());
    tid = -1;
  } catch (...) {
    tid = -1;
  }
  if (nresults >= 1) {
    results[0].kind = WASMTIME_I32;
    results[0].of.i32 = tid;
  }
  return nullptr;
}

void print_loaded_lib() {
  Dl_info info;
  memset(&info, 0, sizeof(info));
  if (dladdr(reinterpret_cast<void *>(&wasmtime_module_new), &info) != 0 &&
      info.dli_fname != nullptr) {
    fprintf(stderr, "loaded-lib=%s\n", info.dli_fname);
  } else {
    fprintf(stderr, "loaded-lib=unknown\n");
  }
}

void print_module_surface(wasmtime_module_t *module, bool *imports_thread_spawn,
                          int *shared_memories, int *plain_memories) {
  wasm_importtype_vec_t imports;
  wasmtime_module_imports(module, &imports);
  for (size_t i = 0; i < imports.size; i++) {
    const wasm_importtype_t *item = imports.data[i];
    std::string module_name = name_of(wasm_importtype_module(item));
    std::string field = name_of(wasm_importtype_name(item));
    const wasm_externtype_t *ty = wasm_importtype_type(item);
    wasm_externkind_t kind = wasm_externtype_kind(ty);
    if (module_name == "wasi" && field == "thread-spawn") {
      *imports_thread_spawn = true;
    }
      if (kind == WASM_EXTERN_MEMORY) {
      const wasm_memorytype_t *memory = wasm_externtype_as_memorytype_const(ty);
      bool shared = wasmtime_memorytype_isshared(memory);
      uint64_t minimum = wasmtime_memorytype_minimum(memory);
      uint64_t maximum = 0;
      bool has_max = wasmtime_memorytype_maximum(memory, &maximum);
      if (shared) {
        (*shared_memories)++;
      } else {
        (*plain_memories)++;
      }
      if (has_max) {
        fprintf(stderr,
                "import module=%s name=%s kind=memory shared=%d min=%llu max=%llu\n",
                module_name.c_str(), field.c_str(), shared ? 1 : 0,
                static_cast<unsigned long long>(minimum),
                static_cast<unsigned long long>(maximum));
      } else {
        fprintf(stderr, "import module=%s name=%s kind=memory shared=%d min=%llu max=none\n",
                module_name.c_str(), field.c_str(), shared ? 1 : 0,
                static_cast<unsigned long long>(minimum));
      }
    } else {
      fprintf(stderr, "import module=%s name=%s kind=%s\n", module_name.c_str(),
              field.c_str(), extern_kind_name(kind));
    }
  }
  wasm_importtype_vec_delete(&imports);

  wasm_exporttype_vec_t exports;
  wasmtime_module_exports(module, &exports);
  for (size_t i = 0; i < exports.size; i++) {
    const wasm_exporttype_t *item = exports.data[i];
    std::string field = name_of(wasm_exporttype_name(item));
    if (field == "_start" || field == "memory" || field == "wasi_thread_start") {
      const wasm_externtype_t *ty = wasm_exporttype_type(item);
      fprintf(stderr, "export name=%s kind=%s\n", field.c_str(),
              extern_kind_name(wasm_externtype_kind(ty)));
    }
  }
  wasm_exporttype_vec_delete(&exports);
}

std::vector<Owned<wasmtime_sharedmemory_t, delete_memory>> define_shared_memories(
    wasm_engine_t *engine, wasmtime_linker_t *linker, wasmtime_context_t *context,
    wasmtime_module_t *module) {
  std::vector<Owned<wasmtime_sharedmemory_t, delete_memory>> memories;
  wasm_importtype_vec_t imports;
  wasmtime_module_imports(module, &imports);
  for (size_t i = 0; i < imports.size; i++) {
    const wasm_importtype_t *item = imports.data[i];
    const wasm_externtype_t *ty = wasm_importtype_type(item);
    if (wasm_externtype_kind(ty) != WASM_EXTERN_MEMORY) {
      continue;
    }
    const wasm_memorytype_t *memory_type = wasm_externtype_as_memorytype_const(ty);
    if (!wasmtime_memorytype_isshared(memory_type)) {
      wasm_importtype_vec_delete(&imports);
      fail("memory import is not shared; wasi-threads requires shared memory");
    }
    wasmtime_sharedmemory_t *created = nullptr;
    check(wasmtime_sharedmemory_new(engine, memory_type, &created),
          "wasmtime_sharedmemory_new");
    memories.emplace_back(created);
    wasmtime_extern_t extern_item;
    memset(&extern_item, 0, sizeof(extern_item));
    extern_item.kind = WASMTIME_EXTERN_SHAREDMEMORY;
    extern_item.of.sharedmemory = created;
    std::string module_name = name_of(wasm_importtype_module(item));
    std::string field = name_of(wasm_importtype_name(item));
    check(wasmtime_linker_define(linker, context, module_name.data(), module_name.size(),
                                 field.data(), field.size(), &extern_item),
          "define shared memory");
  }
  wasm_importtype_vec_delete(&imports);
  return memories;
}

bool stop_is_success(const Stop &stop) {
  return stop.kind == StopKind::Returned ||
         (stop.kind == StopKind::Exit && stop.code == 0);
}

int run(const Options &opt) {
  print_loaded_lib();
  fprintf(stderr, "mode=%s setting=%s has_lse=true strategy=cranelift\n",
          opt.mode.c_str(), opt.setting.c_str());
  fprintf(stderr, "v4-policy=%s v4-rule-version=1\n", opt.v4_policy.c_str());

  wasm_config_t *config = wasm_config_new();
  if (config == nullptr) {
    fail("wasm_config_new failed");
  }
  wasmtime_config_strategy_set(config, WASMTIME_STRATEGY_CRANELIFT);
  wasmtime_config_wasm_threads_set(config, true);
  wasmtime_config_shared_memory_set(config, true);
  wasmtime_config_cranelift_flag_set(config, "has_lse", "true");
  wasmtime_config_cranelift_flag_set(config, "aarch64_mitigation", opt.setting.c_str());
  wasmtime_config_cranelift_flag_set(config, "v4_fence_policy", opt.v4_policy.c_str());
  wasmtime_config_cranelift_flag_set(config, "v4_fence_rule_version", "1");

  Owned<wasm_engine_t, delete_engine> engine(wasm_engine_new_with_config(config));
  if (engine.get() == nullptr) {
    fail("wasm_engine_new_with_config failed");
  }
  fprintf(stderr, "engine-pulley=%d\n", wasmtime_engine_is_pulley(engine.get()) ? 1 : 0);

  std::ifstream input(opt.wasm_path, std::ios::binary);
  if (!input) {
    fail("failed to open wasm: " + opt.wasm_path);
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  wasmtime_module_t *module_raw = nullptr;
  check(wasmtime_module_new(engine.get(), bytes.data(), bytes.size(), &module_raw),
        "wasmtime_module_new");
  Owned<wasmtime_module_t, delete_module> module(module_raw);

  bool imports_thread_spawn = false;
  int shared_memories = 0;
  int plain_memories = 0;
  print_module_surface(module.get(), &imports_thread_spawn, &shared_memories,
                       &plain_memories);
  if (plain_memories != 0) {
    fail("module imports a non-shared memory");
  }
  if (imports_thread_spawn && shared_memories == 0) {
    fail("wasi.thread-spawn is imported but there is no shared memory import");
  }

  Owned<wasmtime_linker_t, delete_linker> linker(wasmtime_linker_new(engine.get()));
  check(wasmtime_linker_define_wasi(linker.get()), "wasmtime_linker_define_wasi");

  Runtime runtime;
  runtime.opt = &opt;
  runtime.engine = engine.get();
  runtime.module = module.get();

  const char *spawn_module = "wasi";
  const char *spawn_field = "thread-spawn";
  wasm_functype_t *spawn_type =
      wasm_functype_new_1_1(wasm_valtype_new_i32(), wasm_valtype_new_i32());
  check(wasmtime_linker_define_func(linker.get(), spawn_module, strlen(spawn_module),
                                    spawn_field, strlen(spawn_field), spawn_type,
                                    on_thread_spawn, &runtime, nullptr),
        "define wasi.thread-spawn");
  wasm_functype_delete(spawn_type);

  Owned<wasmtime_store_t, delete_store> main_store(
      wasmtime_store_new(engine.get(), nullptr, nullptr));
  if (main_store.get() == nullptr) {
    fail("main wasmtime_store_new failed");
  }
  wasmtime_context_t *main_context = wasmtime_store_context(main_store.get());
  auto memories =
      define_shared_memories(engine.get(), linker.get(), main_context, module.get());

  wasmtime_instance_pre_t *pre_raw = nullptr;
  check(wasmtime_linker_instantiate_pre(linker.get(), module.get(), &pre_raw),
        "wasmtime_linker_instantiate_pre");
  Owned<wasmtime_instance_pre_t, delete_pre> pre(pre_raw);
  runtime.pre = pre.get();

  runtime.entry_ok = runtime.thread_entry_ok();
  if (imports_thread_spawn && !runtime.entry_ok) {
    fail("module imports wasi.thread-spawn but does not export wasi_thread_start(i32,i32)->()");
  }
  fprintf(stderr, "thread-entry=%s shared-memory-imports=%d\n",
          runtime.entry_ok ? "yes" : "no", shared_memories);

  runtime.apply_wasi(main_context);

  Watchdog watchdog(opt.timeout_sec);
  struct JoinGuard {
    Runtime *runtime;
    ~JoinGuard() { runtime->join_workers(); }
  } join_guard{&runtime};
  (void)join_guard;
  wasmtime_instance_t instance;
  wasm_trap_t *trap = nullptr;
  wasmtime_error_t *instantiate_error =
      wasmtime_instance_pre_instantiate(pre.get(), main_context, &instance, &trap);
  Stop main_stop;
  if (!opt.time_exec) {
    if (instantiate_error != nullptr || trap != nullptr) {
      main_stop = classify(instantiate_error, trap);
      main_stop.message = "main instantiate: " + main_stop.message;
    } else {
      main_stop = runtime.call_export(main_context, &instance, "_start", nullptr, 0);
    }
    runtime.join_workers();
  } else {
    // Resolve _start before the clock starts. Delete the export after t1 so
    // that lookup and cleanup stay outside the interval.
    wasmtime_extern_t start_item;
    memset(&start_item, 0, sizeof(start_item));
    bool start_ready = false;
    bool start_owned = false;
    std::string resolve_error;
    if (instantiate_error == nullptr && trap == nullptr) {
      if (!wasmtime_instance_export_get(main_context, &instance, "_start", 6,
                                        &start_item)) {
        resolve_error = "missing export _start";
      } else {
        start_owned = true;
        if (start_item.kind != WASMTIME_EXTERN_FUNC) {
          resolve_error = "export is not a func: _start";
        } else {
          start_ready = true;
        }
      }
    }

    struct timespec t0;
    struct timespec t1;
    memset(&t0, 0, sizeof(t0));
    memset(&t1, 0, sizeof(t1));
    if (clock_gettime(CLOCK_MONOTONIC, &t0) != 0) {
      if (start_owned) {
        wasmtime_extern_delete(&start_item);
      }
      fail("clock_gettime CLOCK_MONOTONIC failed before _start");
    }
    if (instantiate_error != nullptr || trap != nullptr) {
      main_stop = classify(instantiate_error, trap);
      main_stop.message = "main instantiate: " + main_stop.message;
    } else if (!start_ready) {
      main_stop.kind = StopKind::Error;
      main_stop.message = resolve_error;
    } else {
      wasm_trap_t *call_trap = nullptr;
      wasmtime_error_t *call_error = wasmtime_func_call(
          main_context, &start_item.of.func, nullptr, 0, nullptr, 0, &call_trap);
      main_stop = classify(call_error, call_trap);
    }
    runtime.join_workers();
    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
      if (start_owned) {
        wasmtime_extern_delete(&start_item);
      }
      fail("clock_gettime CLOCK_MONOTONIC failed after worker join");
    }
    if (start_owned) {
      wasmtime_extern_delete(&start_item);
    }
    long long exec_ns = (t1.tv_sec - t0.tv_sec) * 1000000000LL +
                        (t1.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "exec_ns=%lld\n", exec_ns);
    fprintf(stderr, "timer=CLOCK_MONOTONIC\n");
    fflush(stderr);
  }

  fprintf(stderr, "guest-stop: kind=%s code=%d", stop_kind_name(main_stop.kind),
          main_stop.code);
  if (!main_stop.message.empty()) {
    fprintf(stderr, " message=%s", main_stop.message.c_str());
  }
  fprintf(stderr, "\n");
  fprintf(stderr, "spawn-count=%d workers-joined=%zu\n", runtime.spawn_count,
          runtime.outcomes.size());

  bool workers_ok = true;
  for (const auto &worker : runtime.outcomes) {
    fprintf(stderr, "worker-stop: tid=%d kind=%s code=%d", worker.tid,
            stop_kind_name(worker.stop.kind), worker.stop.code);
    if (!worker.stop.message.empty()) {
      fprintf(stderr, " message=%s", worker.stop.message.c_str());
    }
    fprintf(stderr, "\n");
    if (!stop_is_success(worker.stop)) {
      workers_ok = false;
    }
  }

  bool ok = stop_is_success(main_stop) && workers_ok;
  fprintf(stderr, "result: %s\n", ok ? "ok" : "fail");
  runtime.pre = nullptr;
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options opt = parse_args(argc, argv);
    return run(opt);
  } catch (const std::exception &ex) {
    fprintf(stderr, "host-error: %s\n", ex.what());
    fprintf(stderr, "result: fail\n");
    return 2;
  } catch (...) {
    fprintf(stderr, "host-error: unknown exception\n");
    fprintf(stderr, "result: fail\n");
    return 2;
  }
}
