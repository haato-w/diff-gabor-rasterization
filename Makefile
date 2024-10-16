.PHONY: build

build:
	docker build -t gs-rasterizer-build . -f ./Dockerfile_GS_rasterizer_on_lab

build-11:
	docker build -t gs-rasterizer-build . -f ./Dockerfile_cuda11

up:
	# "capabilities=compute,utility,graphics" is needed for GUI
	# see https://github.com/graphdeco-inria/gaussian-splatting/issues/267
	docker run --net=host -e DISPLAY=$(DISPLAY) --gpus 'all,"capabilities=compute,utility,graphics"' --name 2d-gabor-rasterizer-build-env -w / -it -v ./:/usr/src -p 40207:40207 gs-rasterizer-build /bin/bash

in:
	#docker start 〜
	#docker exec -it コンテナ名 bash

allow-gui-host:
	xhost + local:

# original rasterizer をbuildするコマンド（以下のサイトを参考）
# https://ya6mablog.com/how-to-use-setup-py/#index_id3
build-2d-gabor-rasterizer:
	python3 setup.py bdist_wheel && \
	echo done build
	# pip install --force-reinstall dist/diff_surfel_rasterization-0.0.1-cp312-cp312-linux_x86_64.whl
	