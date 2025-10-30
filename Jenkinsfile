// Helper functions
def getgputargets() {
    return "gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx1201"
}

def getnavi3xtargets() {
    return "gfx1100;gfx1101"
}

def getnavi4xtargets() {
    return "gfx1201"
}

def rocmnodename(name) {
    def rocmtest_name = "(rocmtest || migraphx)"
    def node_name = "${rocmtest_name}"
    if(name == "fiji") {
        node_name = "${rocmtest_name} && fiji"
    } else if(name == "vega") {
        node_name = "${rocmtest_name} && vega"
    } else if(name == "navi21") {
        node_name = "${rocmtest_name} && navi21"
    } else if(name == "mi100+") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a) && !vm"
    } else if(name == "mi200+") {
        node_name = "${rocmtest_name} && (gfx90a || gfx942) && !vm"
    } else if(name == "cdna") {
        node_name = "${rocmtest_name} && (gfx908 || gfx90a || vega20) && !vm"
    } else if(name == "navi32") {
        node_name = "${rocmtest_name} && gfx1101 && !vm"
    } else if(name == "navi4x") {
        node_name = "gfx1201 && !vm"
    } else if(name == "nogpu") {
        node_name = "${rocmtest_name} && nogpu"
    } else if(name == "onnxrt") {
        node_name = "${rocmtest_name} && onnxrt"
    }
    return node_name
}

def dockerBuildAndTest(String dockerArgs = "", Closure body) {
    sh 'printenv'
    checkout scm
    
    // Unstash and read IMAGE_TAG from Setup stage
    unstash 'image-tag'
    IMAGE_TAG = readFile('image-tag.txt').trim()
    echo "Using IMAGE_TAG: ${IMAGE_TAG}"

    def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3').trim()
    def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3').trim()
    def docker_opts = "--device=/dev/kfd --device=/dev/dri --cap-add SYS_PTRACE -v=${env.WORKSPACE}/../:/workspaces:rw,z"
    docker_opts = docker_opts + " --group-add=${video_id} --group-add=${render_id} ${dockerArgs}"
    echo "Docker flags: ${docker_opts}"
    
    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', 
                                    passwordVariable: 'DOCKERHUB_PASS', 
                                    usernameVariable: 'DOCKERHUB_USER')]) {
        sh "echo \$DOCKERHUB_PASS | docker login --username \$DOCKERHUB_USER --password-stdin"
        sh "docker pull ${env.DOCKER_IMAGE}:${IMAGE_TAG}"
        
        withDockerContainer(image: "${env.DOCKER_IMAGE}:${IMAGE_TAG}", args: docker_opts) {
            timeout(time: 4, unit: 'HOURS') {
                body()
            }
        }
    }
}

def cmakeBuild(Map config = [:]) {
    def compiler = config.get("compiler", "/opt/rocm/llvm/bin/clang++")
    def flags = config.get("flags", "")
    def gpu_debug = config.get("gpu_debug", "0")
    
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

pipeline {
    agent {
        label '(rocmtest || migraphx)'
    }
    
    options { 
        skipDefaultCheckout()
    }
    
    parameters {
        booleanParam(name: 'FORCE_DOCKER_IMAGE_BUILD', defaultValue: false, description: 'Force rebuild of Docker image')
    }
    
    environment {
        DOCKER_IMAGE = 'rocm/migraphx-ci-jenkins-ubuntu'
    }
    
    stages {
        stage('Check image') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', 
                                                    passwordVariable: 'DOCKERHUB_PASS', 
                                                    usernameVariable: 'DOCKERHUB_USER')]) {
                        sh "echo \$DOCKERHUB_PASS | docker login --username \$DOCKERHUB_USER --password-stdin"
                        sh 'printenv'
                        checkout scm

                        // Calculate image tag based on file checksums
                        def imageTag = sh(script: '''#!/bin/bash
                            shopt -s globstar
                            sha256sum **/Dockerfile **/*requirements.txt **/install_prereqs.sh **/rbuild.ini **/test/onnx/.onnxrt-commit | sha256sum | cut -d " " -f 1
                        ''', returnStdout: true).trim()
                        echo "Calculated IMAGE_TAG: ${imageTag}"
                        IMAGE_TAG = imageTag
                        
                        // Write to file and stash for other stages
                        writeFile file: 'image-tag.txt', text: imageTag
                        stash includes: 'image-tag.txt', name: 'image-tag'
                        
                        echo "Set IMAGE_TAG: ${IMAGE_TAG}"
                        env.imageExists = sh(script: "docker manifest inspect ${env.DOCKER_IMAGE}:${IMAGE_TAG}", returnStatus: true) == 0 ? 'true' : 'false'
                    }
                }
            }
        }
    
        stage('Build image') {
            when {
                expression { env.imageExists == 'false' || params.FORCE_DOCKER_IMAGE_BUILD }
            }
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'docker_test_cred', 
                                                    passwordVariable: 'DOCKERHUB_PASS', 
                                                    usernameVariable: 'DOCKERHUB_USER')]) {
                        sh "echo \$DOCKERHUB_PASS | docker login --username \$DOCKERHUB_USER --password-stdin"
                        
                        // Unstash and read IMAGE_TAG from Check image stage
                        unstash 'image-tag'
                        IMAGE_TAG = readFile('image-tag.txt').trim()
                        echo "Using IMAGE_TAG for build: ${IMAGE_TAG}"
                        
                        def builtImage
                        try {
                            sh "docker pull ${env.DOCKER_IMAGE}:latest"
                            builtImage = docker.build("${env.DOCKER_IMAGE}:${IMAGE_TAG}", "--cache-from ${env.DOCKER_IMAGE}:latest .")
                        } catch(Exception ex) {
                            builtImage = docker.build("${env.DOCKER_IMAGE}:${IMAGE_TAG}", "--no-cache .")
                        }
                        builtImage.push("${IMAGE_TAG}")
                        builtImage.push("latest")
                    }
                }
            }
        }

        stage('Parallel Tests') {
            parallel {
                stage('All Targets Release') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def gpu_targets = getgputargets()
                                cmakeBuild(flags: "-DCMAKE_BUILD_TYPE=release -DMIGRAPHX_ENABLE_GPU=On -DMIGRAPHX_ENABLE_CPU=On -DMIGRAPHX_ENABLE_FPGA=On -DGPU_TARGETS='${gpu_targets}'")
                            }
                        }
                    }
                }

                stage('Clang ASAN') {
                    agent {
                        label rocmnodename('nogpu')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def sanitizers = "undefined,address"
                                def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers}"
                                cmakeBuild(
                                    flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'",
                                    compiler: '/usr/bin/clang++-14'
                                )
                            }
                        }
                    }
                }
                
                stage('Clang libstdc++ Debug') {
                    agent {
                        label rocmnodename('nogpu')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def sanitizers = "undefined"
                                def debug_flags = "-g -O2 -fno-omit-frame-pointer -fsanitize=${sanitizers} -fno-sanitize-recover=${sanitizers} -D_GLIBCXX_DEBUG"
                                cmakeBuild(
                                    flags: "-DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_C_API_TEST=Off -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_GPU=Off -DMIGRAPHX_ENABLE_CPU=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'",
                                    compiler: '/usr/bin/clang++-14'
                                )
                            }
                        }
                    }
                }

                stage('HIP Clang Release') {
                    agent {
                        label rocmnodename('mi100+')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def gpu_targets = getgputargets()
                                cmakeBuild(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}'")
                                stash includes: 'build/*.deb', name: 'migraphx-package'
                            }
                        }
                    }
                }
                
                stage('HIP Clang Release Navi32') {
                    agent {
                        label rocmnodename('navi32')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def gpu_targets = getnavi3xtargets()
                                cmakeBuild(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                            }
                        }
                    }
                }
                
                stage('HIP Clang Release Navi4x') {
                    agent {
                        label rocmnodename('navi4x')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                def gpu_targets = getnavi4xtargets()
                                cmakeBuild(flags: "-DCMAKE_BUILD_TYPE=release -DGPU_TARGETS='${gpu_targets}' -DMIGRAPHX_DISABLE_ONNX_TESTS=On")
                            }
                        }
                    }
                }

                stage('HIP RTC Debug') {
                    agent {
                        label rocmnodename('mi200+')
                    }
                    environment {
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                withEnv(['MIGRAPHX_DISABLE_MLIR=1']) {
                                    def sanitizers = "undefined"
                                    def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                    def gpu_targets = getgputargets()
                                    cmakeBuild(
                                        flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DMIGRAPHX_USE_HIPRTC=On -DGPU_TARGETS='${gpu_targets}'",
                                        gpu_debug: true
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
                        HSA_ENABLE_SDMA = '0'
                        CCACHE_COMPRESSLEVEL = '7'
                        CCACHE_DIR = '/workspaces/.cache/ccache'
                    }
                    steps {
                        script {
                            dockerBuildAndTest('') {
                                withEnv(['MIGRAPHX_ENABLE_EXTRA_MLIR=1', 
                                        'MIGRAPHX_MLIR_USE_SPECIFIC_OPS=fused,attention,convolution,dot,convolution_backwards', 
                                        'MIGRAPHX_ENABLE_MLIR_INPUT_FUSION=1', 
                                        'MIGRAPHX_MLIR_ENABLE_SPLITK=1', 
                                        'MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION=1', 
                                        'MIGRAPHX_ENABLE_MLIR_GEG_FUSION=1', 
                                        'MIGRAPHX_ENABLE_SPLIT_REDUCE=1',
                                        'MIGRAPHX_DISABLE_LAYERNORM_FUSION=1']) {
                                    def sanitizers = "undefined"
                                    def debug_flags = "-g -O2 -fsanitize=${sanitizers} -fno-sanitize=vptr,function -fno-sanitize-recover=${sanitizers}"
                                    def gpu_targets = getgputargets()
                                    cmakeBuild(
                                        flags: "-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_BUILD_TYPE=debug -DMIGRAPHX_ENABLE_PYTHON=Off -DMIGRAPHX_ENABLE_MLIR=On -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}' -DCMAKE_C_FLAGS_DEBUG='${debug_flags}' -DGPU_TARGETS='${gpu_targets}'"
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
            environment {
                HSA_ENABLE_SDMA = '0'
                CCACHE_COMPRESSLEVEL = '7'
                CCACHE_DIR = '/workspaces/.cache/ccache'
            }
            steps {
                script {
                    dockerBuildAndTest('-u root') {
                        sh 'rm -rf ./build/*.deb'
                        unstash 'migraphx-package'
                        
                        sh '''
                            apt install half
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
