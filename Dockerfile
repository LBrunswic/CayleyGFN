FROM tensorflow:tf14
ARG GIT_REPO=CayleyGFN
ARG GIT_SERVER=node1/git
ARG GIT_HASH=latest
RUN mkdir TASK
RUN git clone http://$GIT_SERVER/$GIT_REPO TASK
WORKDIR /TASK
RUN git config  --global --add advice.detachedHead false
RUN git config --global --add safe.directory /TASK && if [ $GIT_HASH != 'latest' ]; then git checkout $GIT_HASH;fi
VOLUME /TASK/RESULTS
VOLUME /TASK/LOGS
VOLUME /TASK/MODELS
RUN chown -R 1000:1000 /TASK
USER 1000:1000
CMD python3 docker_main.py