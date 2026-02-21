DOCKER_IMAGE = 'rocm/migraphx-ci-jenkins-ubuntu'
DOCKER_IMAGE_ORT = 'rocm/migraphx-ci-jenkins-ubuntu-ort'

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

def setupdocker(Map conf) {
    def options = conf.get("docker_options", "")
    def ccache = "/workspaces/.cache/ccache"
    
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    env.HSA_ENABLE_SDMA = 0
    
    sh 'printenv'
    
    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
    def docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
    docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id}" + options
    echo "Docker flags: ${docker_opts}"
    
    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
        sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
        sh "docker pull ${DOCKER_IMAGE}:${env.IMAGE_TAG}"
    }
    
    return docker_opts
}

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

def rocmtest = { Map conf = [:], Closure body ->
    def variant = conf.get("variant", env.STAGE_NAME)
    def setup = conf.get("setup", {})

    def docker_args = conf.get("docker_args", "")
    def image = conf.get("image", DOCKER_IMAGE)
    def imageTag = conf.get("imageTag", env.IMAGE_TAG)
    def ccache = "/workspaces/.cache/ccache"
    env.CCACHE_COMPRESSLEVEL = 7
    env.CCACHE_DIR = ccache
    env.HSA_ENABLE_SDMA = 0
    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - ${variant}", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX') {
        def docker_opts
        stage("setup ${variant}") {
            sh 'printenv'
            checkout scm
            setup()

            def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
            def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
            docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
            docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id} "
            echo "Docker flags: ${docker_opts}"

            withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                sh "docker pull ${image}:${imageTag}"
            }
        }

        stage("build ${variant}") {
            withDockerContainer(image: "${image}:${imageTag}", args: docker_opts + docker_args) {
                timeout(time: 4, unit: 'HOURS') {
                    body()
                }
            }
        }
    }
}

def setuppackage = {
    sh 'rm -rf ./build/*.deb'
    unstash 'migraphx-package'
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
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - Check image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Checking image', failureDescription: 'Failed to check image', successDescription: 'Image check succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                            sh 'printenv'
                            checkout scm
                            def calculateImageTagScript = """
                                shopt -s globstar
                                sha256sum Dockerfile **/*requirements.txt **/install_prereqs.sh **/rbuild.ini **/test/onnx/.onnxrt-commit | sha256sum | cut -d " " -f 1
                            """
                            env.IMAGE_TAG = sh(script: "bash -c '${calculateImageTagScript}'", returnStdout: true).trim()
                            env.IMAGE_EXISTS = sh(script: "docker manifest inspect ${DOCKER_IMAGE}:${IMAGE_TAG}", returnStatus: true) == 0 ? 'true' : 'false'
                        }
                    }
                }
            }
        }

        stage('Check ORT image') {
            steps {
                script {
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - Check ORT image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Checking ORT image', failureDescription: 'Failed to check ORT image', successDescription: 'ORT image check succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                            sh 'printenv'
                            checkout scm
                            def calculateOrtImageTagScript = """
                                sha256sum tools/docker/ort.dockerfile test/onnx/.onnxrt-commit tools/build_and_test_onnxrt.sh tools/pai_test_launcher.sh tools/pai_provider_test_launcher.sh | sha256sum | cut -d " " -f 1
                            """
                            env.IMAGE_TAG_ORT = sh(script: "bash -c '${calculateOrtImageTagScript}'", returnStdout: true).trim()
                            env.IMAGE_EXISTS_ORT = sh(script: "docker manifest inspect ${DOCKER_IMAGE_ORT}:${IMAGE_TAG_ORT}", returnStatus: true) == 0 ? 'true' : 'false'
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
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - Build image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Building image', failureDescription: 'Failed to build image', successDescription: 'Image build succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                            checkout scm
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

        stage('Build ORT image') {
            when {
                expression { env.IMAGE_EXISTS_ORT == 'false' || params.FORCE_DOCKER_IMAGE_BUILD }
            }
            steps {
                script {
                    gitStatusWrapper(credentialsId: "${env.migraphx_ci_creds}", gitHubContext: "Jenkins - Build ORT image", account: 'ROCmSoftwarePlatform', repo: 'AMDMIGraphX', description: 'Building ORT image', failureDescription: 'Failed to build ORT image', successDescription: 'ORT image build succeeded') {
                        withCredentials([usernamePassword(credentialsId: 'docker_test_cred', passwordVariable: 'DOCKERHUB_PASS', usernameVariable: 'DOCKERHUB_USER')]) {
                            sh "echo $DOCKERHUB_PASS | docker login --username $DOCKERHUB_USER --password-stdin"
                            checkout scm
                            def builtOrtImage

                            try {
                                sh "docker pull ${DOCKER_IMAGE_ORT}:latest"
                                builtOrtImage = docker.build("${DOCKER_IMAGE_ORT}:${IMAGE_TAG_ORT}", "-f tools/docker/ort.dockerfile --cache-from ${DOCKER_IMAGE_ORT}:latest .")
                            } catch(Exception ex) {
                                builtOrtImage = docker.build("${DOCKER_IMAGE_ORT}:${IMAGE_TAG_ORT}", "-f tools/docker/ort.dockerfile --no-cache .")
                            }
                            builtOrtImage.push("${IMAGE_TAG_ORT}")
                            builtOrtImage.push("latest")
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
                            rocmtest([:]) {
                                cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_FPGA=On -DGPU_TARGETS='${getgputargets()}'")
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
                            rocmtest([:]) {
                                def sanitizers = "undefined,address"
                                def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
                                cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'", compiler: '/usr/bin/clang++-17')
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
                            rocmtest([:]) {
                                def sanitizers = "undefined"
                                def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers} -D_GLIBCXX_DEBUG"
                                cmake_build(flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'", compiler: '/usr/bin/clang++-17')
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
                            rocmtest([:]) {
                                cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getgputargets()}'")
                                stash includes: 'build/*.deb', name: 'migraphx-package'
                            }
                        }
                    }
                }

                //stage('HIP Clang Release Navi32') {
                    //agent {
                        //label rocmnodename('navi32')
                    //}
                    //steps {
                        //script {
                            //rocmtest([:]) {
                                //cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getnavi3xtargets()}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                            //}
                        //}
                    //}
                //}
//
                //stage('HIP Clang Release Navi4x') {
                    //agent {
                        //label rocmnodename('navi4x')
                    //}
                    //steps {
                        //script {
                            //rocmtest([:]) {
                                //cmake_build(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${getnavi4xtargets()}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                            //}
                        //}
                    //}
                //}

                stage('HIP RTC Debug') {
                    agent {
                        label rocmnodename('mi200+')
                    }
                    environment {
                        // Disable MLIR since it doesnt work with all ub sanitizers
                        MIGRAPHX_DISABLE_MLIR = '1'
                    }
                    steps {
                        script {
                            rocmtest([:]) {
                                def sanitizers = "undefined"
                                def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                cmake_build(flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${getgputargets()}'", gpu_debug: '1')
                            }
                        }
                    }
                }

                stage('MLIR Debug') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    environment {
                        // Since the purpose of this run is to verify all things MLIR supports,
                        // enabling all possible types of offloads
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
                            rocmtest([:]) {
                                // Note: the -fno-sanitize= is copied from upstream LLVM_UBSAN_FLAGS.
                                def sanitizers = "undefined"
                                def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                cmake_build(flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${getgputargets()}'")
                            }
                        }
                    }
                }
            }
        }

        stage('ONNX Runtime Tests') {
            parallel {
                stage('ONNX Runtime Tests') {
                    agent {
                        label rocmnodename('onnxrt')
                    }
                    steps {
                        script {
                            rocmtest(setup: setuppackage, docker_args: '-u root', image: DOCKER_IMAGE_ORT, imageTag: env.IMAGE_TAG_ORT) {
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
                }
            }
        }
    }
}
