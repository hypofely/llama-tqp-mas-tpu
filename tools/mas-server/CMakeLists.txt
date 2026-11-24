find_package(Threads REQUIRED)

llama_add_compile_flags()

add_executable(mas-server
    mas-server.cpp
)

install(TARGETS mas-server RUNTIME)

target_include_directories(mas-server PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(mas-server PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(mas-server PRIVATE ${CMAKE_SOURCE_DIR}/vendor/nlohmann)
target_include_directories(mas-server PRIVATE ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_link_libraries(mas-server PRIVATE llama common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(mas-server PRIVATE cxx_std_17)
