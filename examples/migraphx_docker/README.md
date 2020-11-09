# Building the MIGraphX Dockerfile

You can build the provided dockerfile with the following command:

    docker build -f Dockerfile.hip-clang -t migraphx .

To run the docker image:

    docker run --device='/dev/kfd' --device='/dev/dri' -v=`pwd`:/data -w /data --group-add video -it migraphx