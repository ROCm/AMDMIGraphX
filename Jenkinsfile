
// def rocmtestnode(variant, name, body, args, pre) {
def rocmtestnode(Map conf) {
    def variant = conf.get("variant")
    def name = conf.get("node")
    def body = conf.get("body")
    def docker_args = conf.get("docker_args", "")
    def docker_build_args = conf.get("docker_build_args", "")
    def pre = conf.get("pre", {})
    def image = 'migraphxlib'
    def cmake_build = { compiler, flags ->
        def cmd = """
            env
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
            CXX=${compiler} CXXFLAGS='-Werror -Wno-fallback' cmake ${flags} .. 
            CTEST_PARALLEL_LEVEL=32 make -j\$(nproc) generate all doc package check
        """
        echo cmd
        sh cmd
        if (compiler == "hcc") {
            // Only archive from master or develop
            if (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master") {
                archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
            }
        }
    }
    node(name) {
        withEnv(['HSA_ENABLE_SDMA=0', 'MIOPEN_DEBUG_GCN_ASM_KERNELS=0']) {
            stage("checkout ${variant}") {
                checkout scm
            }
            gitStatusWrapper(credentialsId: '7126e5fe-eb51-4576-b52b-9aaf1de8f0fd', gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX') {
                pre()
                stage("image ${variant}") {
                    try {
                        docker.build("${image}", "${docker_build_args} .")
                    } catch(Exception ex) {
                        docker.build("${image}", "${docker_build_args} --no-cache .")

                    }
                }
                withDockerContainer(image: image, args: "--device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE ${docker_args}") {
                    timeout(time: 2, unit: 'HOURS') {
                        body(cmake_build)
                    }
                }
            }
        }
    }
}
def rocmtest(m) {
    def builders = [:]
    m.each { e ->
        def label = e.key;
        def action = e.value;
        builders[label] = {
            action(label)
        }
    }
    parallel builders
}

def rocmnodename(name) {
    def rocmtest_name = "(rocmtest || migraphx)"
    def node_name = "${rocmtest_name}"
    if(name == "fiji") {
        node_name = "${rocmtest_name} && fiji";
    } else if(name == "vega") {
        node_name = "${rocmtest_name} && vega";
    }
    return node_name
}

def rocmnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), body: body)
    }
}

def rocmhipclangnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), docker_build_args: '-f hip-clang.docker', body: body)
    }
}

// Static checks
rocmtest format: rocmnode('rocmtest') { cmake_build ->
    stage('Format') {
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
            find . -iname \'*.py\' \
            | grep -v 'build/'  \
            | xargs -n 1 -P 1 -I{} -t sh -c \'yapf {} | diff - {}\'
        '''
    }
}, clang_debug: rocmnode('vega') { cmake_build ->
    stage('Clang Debug') {
        // TODO: Enable integer
        def sanitizers = "undefined"
        def debug_flags = "-O2 -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        cmake_build("hcc", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'")
    }
}, clang_release: rocmnode('vega') { cmake_build ->
    stage('Clang Release') {
        cmake_build("hcc", "-DCMAKE_BUILD_TYPE=release")
        stash includes: 'build/*.deb', name: 'migraphx-package'
    }
}, hip_clang_release: rocmhipclangnode('vega') { cmake_build ->
    stage('Hip Clang Release') {
        cmake_build("/opt/rocm/llvm/bin/clang++", "-DCMAKE_BUILD_TYPE=release")
        // stash includes: 'build/*.deb', name: 'migraphx-package'
    }
}, hip_clang_tidy: rocmhipclangnode('rocmtest') { cmake_build ->
    stage('Hip Clang Tidy') {
        sh '''
            rm -rf build
            mkdir build
            cd build
            CXX=/opt/rocm/llvm/bin/clang++ cmake .. 
            make -j$(nproc) -k analyze
        '''
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
        def debug_flags = "-g -fno-omit-frame-pointer -fsanitize-address-use-after-scope -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
        cmake_build("g++-7", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_PYTHON=Off ${cmake_linker_flags} -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'")

    }
}, codecov: rocmnode('rocmtest') { cmake_build ->
    stage('GCC 7 Codecov') {
        def linker_flags = '-fuse-ld=gold'
        def cmake_linker_flags = "-DCMAKE_EXE_LINKER_FLAGS='${linker_flags}' -DCMAKE_SHARED_LINKER_FLAGS='${linker_flags}'"
        def debug_flags = "-g -fprofile-arcs -ftest-coverage -fno-omit-frame-pointer"
        cmake_build("g++-7", "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_PYTHON=Off ${cmake_linker_flags} -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'")

    }
    stage('Codecov') {
        env.CODECOV_TOKEN="8545af1c-f90b-4345-92a5-0d075503ca56"
        sh '''
            cd build
            lcov --directory . --capture --output-file $(pwd)/coverage.info
            lcov --remove $(pwd)/coverage.info '/usr/*' --output-file $(pwd)/coverage.info
            lcov --list $(pwd)/coverage.info
            curl -s https://codecov.io/bash | bash
            echo "Uploaded"
        '''
    }
}

def onnxnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), docker_args: '-u root', body: body, pre: {
            sh 'rm -rf ./build/*.deb'
            unstash 'migraphx-package' 
        })
    }
}

rocmtest onnx: onnxnode('rocmtest') { cmake_build ->
    stage("Onnx runtime") {
        sh '''
            ls -lR
            dpkg -i --force-depends ./build/*.deb
            cd /onnxruntime && ./build_and_test_onnxrt.sh
        '''
    }
}
