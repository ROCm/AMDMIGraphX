import org.jenkinsci.plugins.pipeline.modeldefinition.Utils

def DOCKER_IMAGE = 'rocm/migraphx-ci-jenkins-ubuntu'

def getgputargets() {
    targets="gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1201"
    return targets
}

def getnavi3xtargets() {
    targets="gfx1100;gfx1101"
    return targets
}

def getnavi4xtargets() {
    targets="gfx1201"
    return targets
}

def rocmtestnode(Map conf) {
    def variant = conf.get("variant")
    def name = conf.get("node")
    def body = conf.get("body")
    def docker_args = conf.get("docker_args", "")
    def docker_build_args = conf.get("docker_build_args", "")
    def pre = conf.get("pre", {})
    def ccache = "/workspaces/.cache/ccache"
    def image = 'migraphxlib'
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    def cmake_build = { bconf ->
        def compiler = bconf.get("compiler", "/opt/rocm/llvm/bin/clang++")
        def flags = bconf.get("flags", "")
        def gpu_debug = bconf.get("gpu_debug", "0")
        def cmd = """
            ulimit -c unlimited
            echo "leak:dnnl::impl::malloc" > suppressions.txt
            echo "leak:libtbb.so" >> suppressions.txt
            cat suppressions.txt
            export LSAN_OPTIONS="suppressions=\$(pwd)/suppressions.txt"
            export ASAN_OPTIONS="detect_container_overflow=0"
            export MIGRAPHX_GPU_DEBUG=${gpu_debug}
            export CXX=${compiler}
            export CXXFLAGS='-Werror'
            rocminfo
            env
            rm -rf build
            mkdir build
            cd build
            cmake -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DBUILD_DEV=On -DCMAKE_EXECUTE_PROCESS_COMMAND_ECHO=STDOUT -DMIGRAPHX_DISABLE_VIRTUAL_ENV=ON ${flags} ..
            git diff
            git diff-index --quiet HEAD || (echo "Git repo is not clean after running cmake." && exit 1)
            make -j\$(nproc) generate VERBOSE=1
            git diff
            git diff-index --quiet HEAD || (echo "Generated files are different. Please run make generate and commit the changes." && exit 1)
            make -j\$(nproc) all package check VERBOSE=1
            md5sum ./*.deb
        """
        echo cmd
        sh cmd
        // Only archive from master or develop
        if (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master") {
            archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        }
    }
    node(name) {
        withEnv(['HSA_ENABLE_SDMA=0']) {
            stage("checkout ${variant}") {
                sh 'printenv'
                checkout scm
            }

            def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
            def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
            def docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
            docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id} "
            echo "Docker flags: ${docker_opts}"

            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX') {
                withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                    sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                    pre()
                    sh "docker pull ${DOCKER_IMAGE}:${env.IMAGE_TAG}"
                    withDockerContainer(image: "${DOCKER_IMAGE}:${env.IMAGE_TAG}", args: docker_opts + docker_args) {
                        timeout(time: 4, unit: 'HOURS') {
                            body(cmake_build)
                        }
                    }
                }
            }
        }
    }
}

def rocmnodename(name) {
    def rocmtest_name = "(rocmtest || migraphx)"
    def node_name = "${rocmtest_name}"
    if(name == "fiji") {
        node_name = "${rocmtest_name} && fiji";
    } else if(name == "vega") {
        node_name = "${rocmtest_name} && vega";
    } else if(name == "navi21") {
        node_name = "${rocmtest_name} && navi21";
    } else if(name == "mi100+") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a) && !vm";
    } else if(name == "mi200+") {
        node_name = "${rocmtest_name} && (gfx90a || gfx942) && !vm";
    } else if(name == "cdna") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a || vega20) && !vm";
    } else if(name == "navi32") {
        node_name = "${rocmtest_name} && gfx1101 && !vm";
    } else if(name == "navi4x") {
        node_name = "gfx1201 && !vm";
    } else if(name == "nogpu") {
        node_name = "${rocmtest_name} && nogpu";
    } else if(name == "onnxrt") {
        node_name = "${rocmtest_name} && onnxrt";
    }
    return node_name
}

def rocmnode(name, body) {
    return { label ->
        rocmtestnode(variant: label, node: rocmnodename(name), body: body)
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

pipeline {
    agent none
    
    parameters {
        booleanParam(name: 'FORCE_DOCKER_IMAGE_BUILD', defaultValue: false)
    }
    
    stages {
        stage('Check and Build Image') {
            agent {
                label "(rocmtest || migraphx)"
            }
            stages {
                stage('Check image') {
                    steps {
                        script {
                            withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                                sh "echo \$DOCKERHUB_PASS | docker login --username \$DOCKERHUB_USER --password-stdin"
                                sh 'printenv'
                                checkout scm
                                def calculateImageTagScript = """
                                    shopt -s globstar
                                    sha256sum **/Dockerfile **/*requirements.txt **/install_prereqs.sh **/rbuild.ini **/test/onnx/.onnxrt-commit | sha256sum | cut -d " " -f 1
                                """
                                env.IMAGE_TAG = sh(script: "bash -c '${calculateImageTagScript}'", returnStdout: true).trim()
                                env.IMAGE_EXISTS = sh(script: "docker manifest inspect ${DOCKER_IMAGE}:${IMAGE_TAG}", returnStatus: true) == 0 ? 'true' : 'false'
                            }
                        }
                    }
                }
                
                stage('Build image') {
                    when {
                        expression { env.IMAGE_EXISTS == 'false' || params.FORCE_DOCKER_IMAGE_BUILD }
                    }
                    steps {
                        script {
                            withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                                sh "echo \$DOCKERHUB_PASS | docker login --username \$DOCKERHUB_USER --password-stdin"
                                def builtImage
                                try {
                                    sh "docker pull ${DOCKER_IMAGE}:latest"
                                    builtImage = docker.build("${DOCKER_IMAGE}:${IMAGE_TAG}", "--cache-from ${DOCKER_IMAGE}:latest .")
                                } catch(Exception ex) {
                                    builtImage = docker.build("${DOCKER_IMAGE}:${IMAGE_TAG}", " --no-cache .")
                                }
                                builtImage.push("${IMAGE_TAG}")
                                builtImage.push("latest")
                            }
                        }
                    }
                }
            }
        }
        
        stage('Parallel Tests') {
            parallel {
                stage('clang_debug') {
                    steps {
                        script {
                            def testNode = rocmnode('mi200+') { cmake_build ->
                                stage('hipRTC Debug') {
                                    // Disable MLIR since it doesnt work with all ub sanitizers
                                    withEnv(['MIGRAPHX_DISABLE_MLIR=1']) {
                                        def sanitizers = "undefined"
                                        def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                        def gpu_targets = getgputargets()
                                        cmake_build(flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${gpu_targets}'", gpu_debug: true)
                                    }
                                }
                            }
                            testNode('clang_debug')
                        }
                    }
                }
                
                stage('clang_release') {
                    steps {
                        script {
                            def testNode = rocmnode('mi100+') { cmake_build ->
                                stage('Hip Clang Release') {
                                    def gpu_targets = getgputargets()
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}'")
                                    stash includes: 'build/*.deb', name: 'migraphx-package'
                                }
                            }
                            testNode('clang_release')
                        }
                    }
                }
                
                stage('all_targets_debug') {
                    steps {
                        script {
                            def testNode = rocmnode('mi100+') { cmake_build ->
                                stage('All targets Release') {
                                    def gpu_targets = getgputargets()
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_FPGA=On -DGPU_TARGETS='${gpu_targets}'")
                                }
                            }
                            testNode('all_targets_debug')
                        }
                    }
                }
                
                stage('mlir_debug') {
                    steps {
                        script {
                            def testNode = rocmnode('mi100+') { cmake_build ->
                                stage('MLIR Debug') {
                                    withEnv(['MIGRAPHX_ENABLE_EXTRA_MLIR=1', 'MIGRAPHX_MLIR_USE_SPECIFIC_OPS=fused,attention,convolution,dot,convolution_backwards', 'MIGRAPHX_ENABLE_MLIR_INPUT_FUSION=1', 'MIGRAPHX_MLIR_ENABLE_SPLITK=1', 'MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION=1', 'MIGRAPHX_ENABLE_MLIR_GEG_FUSION=1', 'MIGRAPHX_ENABLE_SPLIT_REDUCE=1','MIGRAPHX_DISABLE_LAYERNORM_FUSION=1']) {
                                        def sanitizers = "undefined"
                                        // Note: the -fno-sanitize= is copied from upstream LLVM_UBSAN_FLAGS.
                                        def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                        def gpu_targets = getgputargets()
                                        // Since the purpose of this run verify all things MLIR supports,
                                        // enabling all possible types of offloads
                                        cmake_build(flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${gpu_targets}'")
                                    }
                                }
                            }
                            testNode('mlir_debug')
                        }
                    }
                }
                
                stage('clang_release_navi') {
                    steps {
                        script {
                            def testNode = rocmnode('navi32') { cmake_build ->
                                stage('HIP Clang Release Navi32') {
                                    def gpu_targets = getnavi3xtargets()
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                                }
                            }
                            testNode('clang_release_navi')
                        }
                    }
                }
                
                stage('clang_asan') {
                    steps {
                        script {
                            def testNode = rocmnode('nogpu') { cmake_build ->
                                stage('Clang ASAN') {
                                    def sanitizers = "undefined,address"
                                    def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'", compiler:'/usr/bin/clang++-14')
                                }
                            }
                            testNode('clang_asan')
                        }
                    }
                }
                
                stage('debub_libstdcxx') {
                    steps {
                        script {
                            def testNode = rocmnode('nogpu') { cmake_build ->
                                stage('Debug libstdc++') {
                                    def sanitizers = "undefined"
                                    def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers} -D_GLIBCXX_DEBUG"
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'", compiler:'/usr/bin/clang++-14')
                                }
                            }
                            testNode('debub_libstdcxx')
                        }
                    }
                }
                
                stage('clang_release_navi4') {
                    steps {
                        script {
                            def testNode = rocmnode('navi4x') { cmake_build ->
                                stage('HIP Clang Release Navi4x') {
                                    def gpu_targets = getnavi4xtargets()
                                    cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                                }
                            }
                            testNode('clang_release_navi4')
                        }
                    }
                }
            }
        }
        
        stage('ONNX Tests') {
            steps {
                script {
                    def testNode = onnxnode('onnxrt') { cmake_build ->
                        stage("Onnx runtime") {
                            sh '''
                                apt install half
                                #ls -lR
                                md5sum ./build/*.deb
                                dpkg -i ./build/*.deb
                                env
                                cd /onnxruntime && ./build_and_test_onnxrt.sh
                            '''
                        }
                    }
                    testNode('onnx')
                }
            }
        }
    }
}
