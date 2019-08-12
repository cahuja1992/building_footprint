if docker run --rm -it --runtime=nvidia \
-e CUDA_VISIBLE_DEVICES=1 \
--expose=22 \
-v `pwd`:/ws \
-v /home/antpc/visteon/deepSpeech/docker_files:/docker_files \
-v /media/antpc/hdd/:/hdd \
-p 6006:6006 \
--workdir /ws \
--name deepspeech_specaugment \
--ipc host \
$1 bash
then
    :
else
    docker exec -it deepspeech_specaugment bash
fi
