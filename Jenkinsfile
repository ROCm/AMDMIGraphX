DOCKER_IMAGE = 'rocm/migraphx-ci-jenkins-ubuntu'

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

def setuprocmtest() {
    def ccache = "/workspaces/.cache/ccache"
    
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    env.HSA_ENABLE_SDMA = 0
    
    sh 'printenv'
    checkout scm
    
    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
    def docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
    docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id}"
    echo "Docker flags: ${docker_opts}"
    
    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
        sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
        sh "docker pull ${DOCKER_IMAGE}:${env.IMAGE_TAG}"
    }
    
    return docker_opts
}

def buildrocmtest(Map conf, String docker_opts) {
    def flags = conf.get("flags", "")
    def compiler = conf.get("compiler", "/opt/rocm/llvm/bin/clang++")
    def gpu_debug = conf.get("gpu_debug", "0")
    
    withDockerContainer(image: "${DOCKER_IMAGE}:${env.IMAGE_TAG}", args: docker_opts) {
        timeout(time: 4, unit: 'HOURS') {
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
    }
}

def setuponnxtest() {
    def ccache = "/workspaces/.cache/ccache"
    
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    env.HSA_ENABLE_SDMA = 0
    
    sh 'printenv'
    sh 'rm -rf ./build/*.deb'
    unstash 'migraphx-package'
    checkout scm
    
    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
    def docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
    docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id} -u root"
    echo "Docker flags: ${docker_opts}"
    
    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
        sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
        sh "docker pull ${DOCKER_IMAGE}:${env.IMAGE_TAG}"
    }
    
    return docker_opts
}

def buildonnxtest(String docker_opts) {
    withDockerContainer(image: "${DOCKER_IMAGE}:${env.IMAGE_TAG}", args: docker_opts) {
        timeout(time: 4, unit: 'HOURS') {
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
}

pipeline {
    agent {
        label "(rocmtest || migraphx)"
    }

    options {
        skipDefaultCheckout()
    }

    parameters {
        booleanParam(name: 'FORCE_DOCKER_IMAGE_BUILD', defaultValue: false)
    }

    stages {
        stage('Check image') {
            steps {
                script {
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Check image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Checking image', failureDescription: 'Failed to check image', successDescription: 'Image check succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
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
        }

        stage('Build image') {
            when {
                expression { env.IMAGE_EXISTS == 'false' || params.FORCE_DOCKER_IMAGE_BUILD }
            }
            steps {
                script {
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Build image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Building image', failureDescription: 'Failed to build image', successDescription: 'Image build succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
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

        stage('Tests') {
            parallel {
                stage('All Targets Release') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "All Targets Release", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('All Targets Release - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('All Targets Release - Build') {
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_FPGA=On -DGPU_TARGETS='${getgputargets()}'",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('Clang ASAN') {
                    agent {
                        label rocmnodename('nogpu')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Clang ASAN", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('Clang ASAN - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('Clang ASAN - Build') {
                                    def sanitizers = "undefined,address"
                                    def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'",
                                         compiler: '/usr/bin/clang++-14',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('Clang libstdc++ Debug') {
                    agent {
                        label rocmnodename('nogpu')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Clang libstdc++ Debug", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('Clang libstdc++ Debug - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('Clang libstdc++ Debug - Build') {
                                    def sanitizers = "undefined"
                                    def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers} -D_GLIBCXX_DEBUG"
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'",
                                         compiler: '/usr/bin/clang++-14',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('HIP Clang Release') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "HIP Clang Release", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('HIP Clang Release - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('HIP Clang Release - Build') {
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getgputargets()}'",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                    stash includes: 'build/*.deb', name: 'migraphx-package'
                                }
                            }
                        }
                    }
                }

                stage('HIP Clang Release Navi32') {
                    agent {
                        label rocmnodename('navi32')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "HIP Clang Release Navi32", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('HIP Clang Release Navi32 - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('HIP Clang Release Navi32 - Build') {
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getnavi3xtargets()}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('HIP Clang Release Navi4x') {
                    agent {
                        label rocmnodename('navi4x')
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "HIP Clang Release Navi4x", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('HIP Clang Release Navi4x - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('HIP Clang Release Navi4x - Build') {
                                    buildrocmtest(
                                        [flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getnavi4xtargets()}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('HIP RTC Debug') {
                    agent {
                        label rocmnodename('mi200+')
                    }
                    environment {
                        MIGRAPHX_DISABLE_MLIR = '1'
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "HIP RTC Debug", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('HIP RTC Debug - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('HIP RTC Debug - Build') {
                                    // Disable MLIR since it doesnt work with all ub sanitizers
                                    def sanitizers = "undefined"
                                    def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                    buildrocmtest(
                                        [flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${getgputargets()}'",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '1'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }

                stage('MLIR Debug') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    environment {
                        MIGRAPHX_ENABLE_EXTRA_MLIR = '1'
                        MIGRAPHX_MLIR_USE_SPECIFIC_OPS = 'fused,attention,convolution,dot,convolution_backwards'
                        MIGRAPHX_ENABLE_MLIR_INPUT_FUSION = '1'
                        MIGRAPHX_MLIR_ENABLE_SPLITK = '1'
                        MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION = '1'
                        MIGRAPHX_ENABLE_MLIR_GEG_FUSION = '1'
                        MIGRAPHX_ENABLE_SPLIT_REDUCE = '1'
                        MIGRAPHX_DISABLE_LAYERNORM_FUSION = '1'
                    }
                    steps {
                        script {
                            gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "MLIR Debug", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                                stage('MLIR Debug - Setup') {
                                    env.DOCKER_OPTS = setuprocmtest()
                                }
                                stage('MLIR Debug - Build') {
                                    def sanitizers = "undefined"
                                    // Note: the -fno-sanitize= is copied from upstream LLVM_UBSAN_FLAGS.
                                    def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                    // Since the purpose of this run is to verify all things MLIR supports,
                                    // enabling all possible types of offloads
                                    buildrocmtest(
                                        [flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${getgputargets()}'",
                                         compiler: '/opt/rocm/llvm/bin/clang++',
                                         gpu_debug: '0'],
                                        env.DOCKER_OPTS
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        stage('ONNX Runtime Tests') {
            agent {
                label rocmnodename('onnxrt')
            }
            steps {
                script {
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "ONNX Runtime Tests", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Running stage', failureDescription: 'Failed stage', successDescription: 'Stage succeeded') {
                        stage('ONNX Runtime Tests - Setup') {
                            env.DOCKER_OPTS = setuponnxtest()
                        }
                        stage('ONNX Runtime Tests - Build') {
                            buildonnxtest(env.DOCKER_OPTS)
                        }
                    }
                }
            }
        }
    }
}
