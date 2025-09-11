
> Copyright
Jelen text file a Budapesti Műszaki és Gazdaságtudományi Egyetemen tartott "Deep Learning a gyakorlatban Python és LUA alapon" tantárgy segédanyagaként készült. 
A tantárgy honlapja: http://smartlab.tmit.bme.hu/oktatas-deep-learning
Deep Learning kutatás: http://smartlab.tmit.bme.hu/deep-learning
Az anyag bármely részének újra felhasználása, publikálása csak a szerzők írásos beleegyezése esetén megengedett.

> 2020 (c) Gyires-Tóth Bálint (toth.b kukac tmit pont bme pont hu), Zainkó Csaba


***********************************************************

Installálni a "docker"-t, GPU-hoz az "nvidia-docker"-t kell.

Containerek listája:

	docker ps 
	docker ps -a

Futtatás, példa:

	sudo nvidia-docker run  nvidia/cuda:9.0-base nvidia-smi

Futtatás, példa interaktív módban:

	sudo nvidia-docker run  -it nvidia/cuda:9.0-base bash

Docker image-ek listája:

	sudo docker images

Docker image törlése:

	sudo docker rmi

Container törlése:

	sudo docker rm

	(--rm paraméter a run-nál, futtatás után törli)

Image készítése:

	Dockerfile-ba az alábbiakat bemásolni, content könyvtárban van a saját fájljaink.

	FROM ufoym/deepo:pytorch-py36-cu90
	COPY content /content
	RUN pip install librosa && \
		pip install unidecode &&\
		pip install inflect && \
		pip install tensorboardX &&\
		pip install tensorflow-gpu &&\ 
		pip install matplotlib==2.1.0 &&\
		pip install torch==1.0.0 &&\
		pip install inflect &&\
		pip install scipy &&\
		pip install pillow &&\
		apt-get update && apt-get  install -y \
		nano \
		tmux \
		mc  && \
		rm -rf /var/lib/apt/lists/*
	ENTRYPOINT ["/bin/sh", "-c"]
	WORKDIR "/content/waveglow-master"
	CMD ["/content/waveglow-master/start.sh"] 

	Majd Docker image elkészítése:

	sudo docker build . -t smartlab/vitmav45:test_image

Docker futtatása, interakív módban:

	sudo nvidia-docker run -it --ipc="host" --shm-size 120g --rm --network host -v /home/host/folder:/shared:rw smartlab/vitmav45:test_image bash
