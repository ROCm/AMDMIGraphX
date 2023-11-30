#include <iostream>
#include <string>
#include <vector>

#include <Windows.h>

#include <migraphx/errors.hpp>


int main() { 
	std::vector<char> result;
    HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE std_out = GetStdHandle(STD_OUTPUT_HANDLE);

    if(std_in == INVALID_HANDLE_VALUE)
        MIGRAPHX_THROW("STDIN invalid handle (" + std::to_string(GetLastError()) + ")");
    constexpr std::size_t BUFFER_SIZE = 1024;
    DWORD bytes_read;
    TCHAR buffer[BUFFER_SIZE];
    for(;;)
    {
        BOOL status = ReadFile(std_in, buffer, BUFFER_SIZE, &bytes_read, nullptr);
        if(status == FALSE or bytes_read == 0)
            break;
        DWORD written;
        if(WriteFile(std_out, buffer, bytes_read, &written, nullptr) == FALSE)
            break;
        //result.insert(result.end(), buffer, buffer + bytes_read);
    }
    //std::cout << result.data();
    return 0;
}
