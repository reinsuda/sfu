project('sfu', 'cpp',
  version : '0.1.0',
  default_options : ['warning_level=3', 'cpp_std=c++14']) # 你可以根据实际情况修改C++标准

# 1. 定义头文件包含目录
# 把 coeff 和 common 目录包含进来，这样 src 和 test 里的代码就可以直接 #include
inc_dir = include_directories('coeff', 'common')

# 2. 定义源文件
# 列出 src 目录下的所有 .cpp 文件
sfu_srcs = [
  'src/cos.cpp',
  'src/log2.cpp',
  'src/rcp.cpp',
  'src/rsq.cpp',
  'src/sin.cpp',
  'src/sqrt.cpp'
]

# 3. 编译核心库
# 将 src 目录下的代码编译为一个静态库 (你也可以用 shared_library 或 library)
sfu_lib = static_library('sfu',
  sfu_srcs,
  include_directories : inc_dir
)

# 4. 编译测试程序
# 将 test/test.cpp 编译为可执行文件，并链接上面的 sfu_lib
test_exe = executable('sfu_test',
  'test/test.cpp',
  include_directories : inc_dir,
  link_with : sfu_lib
)

# 5. 注册测试
# 这样你就可以使用 `meson test` 命令来运行它了
test('sfu basic test', test_exe)