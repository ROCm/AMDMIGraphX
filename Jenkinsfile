
def rocmtestnode(variant, name, body) {
    def image = 'migraphxlib'
    def cmake_build = { compiler, flags ->
        def cmd = """
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror -Wno-fallback' cmake ${flags} .. 
            CTEST_PARALLEL_LEVEL=32 make -j32 all doc check
        """
        echo cmd
        sh cmd
    }
    node(name) {
        stage("checkout ${variant}") {
            env.HSA_ENABLE_SDMA=0 
            checkout scm
        }
        stage("image ${variant}") {
            try {
                docker.build("${image}", '.')
            } catch(Exception ex) {
                docker.build("${image}", '--no-cache .')

            }
        }
        withDockerContainer(image: image, args: '--device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE') {
            timeout(time: 1, unit: 'HOURS') {
                body(cmake_build)
            }
        }
    }
}
@NonCPS
def rocmtest(m) {
    def builders = [:]
    for(e in m) {
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

@NonCPS
def rocmnode(name, body) {
    def node_name = 'rocmtest || rocm'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else {
        node_name = name
    }
    return { label ->
        rocmtestnode(label, node_name, body)
    }
}

@NonCPS
def rocmnode(body) {
    rocmnode('rocmtest', body)
}

// Static checks
rocmtest tidy: rocmnode('rocmtest') { cmake_build ->
    stage('Clang Tidy') {
        sh '''
            rm -rf build
            mkdir build
            cd build
            CXX='clang++-5.0' cmake .. 
            make -j8 -k analyze
        '''
    }
}, format: rocmnode('rocmtest') { cmake_build ->
    stage('Clang Format') {
        sh '''
            find . -iname \'*.h\' \
                -o -iname \'*.hpp\' \
                -o -iname \'*.cpp\' \
                -o -iname \'*.h.in\' \
                -o -iname \'*.hpp.in\' \
                -o -iname \'*.cpp.in\' \
                -o -iname \'*.cl\' \
            | grep -v 'build/' \
            | xargs -n 1 -P 1 -I{} -t sh -c \'clang-format-5.0 -style=file {} | diff - {}\'
        '''
    }
}, clang: rocmnode('rocmtest') { cmake_build ->
    stage('Clang Debug') {
        // TODO: Enanle integer
        def sanitizers = "undefined"
        cmake_build("hcc", "-DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='-g -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}'")
    }
    stage('Clang Release') {
        cmake_build("hcc", "-DCMAKE_BUILD_TYPE=release")
    }
}, gcc5: rocmnode('rocmtest') { cmake_build ->
    stage('GCC 5 Debug') {
        cmake_build("g++-5", "-DCMAKE_BUILD_TYPE=debug")
    }
    stage('GCC 5 Release') {
        cmake_build("g++-5", "-DCMAKE_BUILD_TYPE=release")
    }
}, gcc7: rocmnode('rocmtest') { cmake_build ->
    stage('GCC 7 Debug') {
        def linker_flags = '-fuse-ld=gold'
        def cmake_linker_flags = "-DCMAKE_EXE_LINKER_FLAGS='${linker_flags}' -DCMAKE_SHARED_LINKER_FLAGS='${linker_flags}'"
        // TODO: Add bounds-strict
        def sanitizers = "undefined,address"
        cmake_build("g++-7", "-DCMAKE_BUILD_TYPE=debug ${cmake_linker_flags} -DCMAKE_CXX_FLAGS_DEBUG='-g -fno-omit-frame-pointer -fsanitize-address-use-after-scope -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}'")
    }
}
