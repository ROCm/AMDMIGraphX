// RULES:
// 1) Build and cache the docker 
// 2) Be able to restart parts of the pipeline
// 3) Check your targets 
//
// Build Process
//
// HIP Clang Docker --> "all targets", "clang asan", etc 
// ORT Docker       --> "ORT benchmark"
//
// Each docker can be used on any system

def rocmnode(name) {
    return 'rocmtest && (' + name + ')'
}

def getDockerImageName(dockerArgs)
{
    sh "echo ${dockerArgs} > factors.txt"
    def image = "rocm/migraphx-ci-ubuntu"
    sh "md5sum Dockerfile requirements.txt dev-requirements.txt >> factors.txt"
    def docker_hash = sh(script: "md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE != '')
    {
        echo "Overriding the base docker image with ${params.DOCKER_IMAGE_OVERRIDE}"
        image = "${params.DOCKER_IMAGE_OVERRIDE}"
    }
    return image

}

def getDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def gpu_arch = "gfx1030;gfx1100;gfx1101;gfx1102" // prebuilt dockers should have all the architectures enabled so one image can be used for all stages
    def dockerArgs = "--build-arg GPU_TARGETS='${gpu_arch}'"
    echo "Docker Args: ${dockerArgs}"

    def image = getDockerImageName(dockerArgs)

    def dockerImage
    try{
        echo "Pulling down image: ${image}"
        dockerImage = docker.image("${image}")
        dockerImage.pull()
    }
    catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
    catch(Exception ex)
    {
        dockerImage = docker.build("${image}", "${dockerArgs} .")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            dockerImage.push()
        }
    }
    return [dockerImage, image]
}


pipeline {
    agent none
    options {
        parallelsAlwaysFailFast()
    }
    parameters {
        booleanParam(
            name: "BUILD_DOCKER",
            defaultValue: true,
            description: "")
        booleanParam(
            name: "BUILD_STATIC_CHECKS",
            defaultValue: true,
            description: "")
        string(name: "DOCKER_IMAGE_OVERRIDE",
            defaultValue: '',
            description: "")            
    }

    stages{
        stage('Build Docker'){
            when {
                expression { params.BUILD_DOCKER }
            }
            agent{ label rocmnode("nogpu") }
            steps{
                getDockerImage()
            }
        }
        stage("Static checks") {
            parallel{
                stage('Hip Tidy') {
                    agent{ label rocmnode("nogpu") }
                    environment{
                    }
                    steps{
                        sh "echo Hi from Hip Tidy"
                    }
                }
                stage('Clang Format') {
                    agent{ label rocmnode("nogpu") }
                    steps{
                        sh "echo Hi from Clang Format"
                    }
                }
            }
        }
    }
}
