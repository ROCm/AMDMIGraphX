::
:: The MIT License (MIT)
::
:: Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
::
:: Permission is hereby granted, free of charge, to any person obtaining a copy
:: of this software and associated documentation files (the "Software"), to deal
:: in the Software without restriction, including without limitation the rights
:: to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
:: copies of the Software, and to permit persons to whom the Software is
:: furnished to do so, subject to the following conditions:
::
:: The above copyright notice and this permission notice shall be included in
:: all copies or substantial portions of the Software.
::
:: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
:: IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
:: FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
:: AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
:: LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
:: OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
:: THE SOFTWARE.

@echo off
if not defined CLANG_FORMAT_10 (
    set CLANG_FORMAT_10=C:\Program Files\LLVM\10.0\bin\clang-format.exe
)
python --version >nul 2>&1
if errorlevel 1 (
    echo Error^: Python not installed!
    goto :eof
)
for /f "usebackq delims=|" %%f in (`dir /b "%cd%\include"`) do (
    python %cd%\te.py %cd%\include\%%f | "%CLANG_FORMAT_10%" -style=file > %cd%\..\src\include\migraphx\%%f
)
call :api %cd%\api\migraphx.h %cd%\..\src\api\include\migraphx\migraphx.h
echo Finished generating header migraphx.h
call :api %cd%\api\api.cpp %cd%\..\src\api\api.cpp
echo Finished generating source api.h
goto :eof
:api
python %cd%\api.py %cd%\..\src\api\migraphx.py %~1 | "%CLANG_FORMAT_10%" -style=file > %~2
exit /b 0
