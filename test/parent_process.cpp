#include <Windows.h>
#include <iostream>
#include <string>

#include "test.hpp"
#include <migraphx/msgpack.hpp>
#include <migraphx/process.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/errors.hpp>


TEST_CASE(string_data)
{
    auto child_path = migraphx::fs::path{"C:/develop/AMDMIGraphX/build/bin"};

    std::string string_data = "Parent string \0";

    // write string data to child process
    migraphx::process{"test_child.exe"}.cwd(child_path).write(
        [&](auto writer) { migraphx::to_msgpack(string_data, writer); });

    //// parent process read from child stdout
    //std::vector<char> result;
    //HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    //if(std_in == INVALID_HANDLE_VALUE)
    //    MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    //constexpr std::size_t BUFFER_SIZE = 4096;
    //DWORD bytes_read;
    //TCHAR buffer[BUFFER_SIZE];
    //for(;;)
    //{
    //    BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
    //    if(status == FALSE or bytes_read == 0)
    //        break;

    //    result.insert(result.end(), buffer, buffer + bytes_read);
    //}

    //EXPECT(result.data() == string_data); 
}

TEST_CASE(binary_data)
{

    // binary data
    std::vector<char> binary_data = {'B', 'i', 'n', 'a', 'r', 'y'};
    auto child_path = migraphx::fs::path{"C:/develop/AMDMIGraphX/build/bin"};

    // write string data to child process
    migraphx::process{"test_child.exe"}.cwd(child_path).write([&](auto writer) {
        migraphx::to_msgpack(binary_data, writer);
    });

    //// parent process read from child stdout
    //std::vector<char> result;
    //HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    //if(std_in == INVALID_HANDLE_VALUE)
    //    MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    //constexpr std::size_t BUFFER_SIZE = 4096;
    //DWORD bytes_read;
    //TCHAR buffer[BUFFER_SIZE];
    //for(;;)
    //{
    //    BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
    //    if(status == FALSE or bytes_read == 0)
    //        break;

    //    result.insert(result.end(), buffer, buffer + bytes_read);
    //}

    //EXPECT(result.data() == string_data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
 
