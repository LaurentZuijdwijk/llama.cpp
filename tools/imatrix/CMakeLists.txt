set(TARGET llama-imatrix)
add_executable(${TARGET} imatrix.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # AIX's flock() function comes from libbsd.a
    target_link_libraries(${TARGET} PRIVATE -lbsd)
endif()

# Full-Hessian collection (LLAMA_HESSIAN_OUT) accumulates X^T X with cblas_ssyrk.
find_library(LLAMA_IMATRIX_OPENBLAS NAMES openblas libopenblas.so.0 PATHS /usr/lib/x86_64-linux-gnu)
if (NOT LLAMA_IMATRIX_OPENBLAS)
    message(FATAL_ERROR "llama-imatrix Hessian collection needs OpenBLAS (libopenblas.so.0)")
endif()
target_link_libraries(${TARGET} PRIVATE ${LLAMA_IMATRIX_OPENBLAS})
