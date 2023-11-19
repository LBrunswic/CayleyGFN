FROM tensorflow:tf14
ARG GIT_REPO=CayleyGFN
ARG GIT_SERVER=node1/git
ARG GIT_HASH=eaa9ed264494705d5c98dbb47190076dc0e79dfe
RUN mkdir TASK
RUN git clone http://$GIT_SERVER/$GIT_REPO TASK
WORKDIR /TASK
RUN git config  --global --add advice.detachedHead false
RUN git config --global --add safe.directory /TASK && git checkout $GIT_HASH
RUN chown -R 1000:1000 /TASK
VOLUME /TASK/RESULTS
VOLUME /TASK/LOGS
VOLUME /TASK/MODELS
USER 1000:1000
CMD python3 docker_main.py