NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus all
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
# BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) --shm-size 20G
# BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) -v /home/rumon/Desktop/cpdir:/checkpoints --shm-size 20G
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) -v /home/rumon/getart/artifacts:/host_artifacts --shm-size 20G

RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = jaxmarl
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/.

run:
	$(DOCKER_RUN) /bin/bash

# run-overcooked:
# 	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_d1.py +layouts=[cramped_room,new_layout]"

# run-overcooked:
# 	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_d1.py +layout=cramped_room"

run-overcooked:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/D1_ippo_ff_overcooked_v2_random.py"

run-transplant:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/D1_ippo_rnn_overcooked_v2_randomness_self_blind_transplant5.py"


run-sb3:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_sb3.py"


run-mpe:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_mpe_facmac.py"

# run-interactive:
# 	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/human_interacting_with_rl_policy.py"

run-interactive:
	docker run $(RUN_FLAGS) \
		-e DISPLAY=$(DISPLAY) \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		--network host \
		$(IMAGE) \
		/bin/bash -c "python -u baselines/IPPO/human_interacting_with_rl_policy.py"


run-overcooked-rnn:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_rnn_overcooked_v25.py"

run-load:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_load.py"

run-load2:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/convert_model3.py"

# run-overcooked:
# 	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_modular2.py"

run-convert2:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/convert_model3.py"

run-astar:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/astar_ru3.py"

run-atest:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/neural-astar-jax-main/notebooks/test.py"

run-modular:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_modular.py"

run-convert:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/convert_model.py"

run-pretrain:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/get_data_train.py"

run-idea1:
	$(DOCKER_RUN) /bin/bash -c "python -u baselines/IPPO/ippo_ff_overcooked_v2_idea1.py"

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"