#!/bin/bash
if docker run --rm -it \
--security-opt seccomp=unconfined \
-e uid=$UID \
-e LOCAL_USER_ID=`id -u` \
-p 8888:8888 \
-v `pwd`:/ws \
-v `pwd`/entrypoint.sh:/usr/bin/entrypoint.sh \
--name building_footprint \
ahujachirag/fastai 
then
    :
else
    docker exec -it building_footprint
fi
