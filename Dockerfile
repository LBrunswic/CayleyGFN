FROM tensorflow:tf14
ARG GIT_REPO=CayleyGFN
ARG GIT_SERVER=10.0.0.85/git
ARG GIT_HASH=latest
RUN mkdir TASK
RUN git clone http://$GIT_SERVER/$GIT_REPO TASK
WORKDIR /TASK
RUN mkdir LOGS
RUN mkdir RESULTS
RUN mkdir MODELS
RUN mkdir HP
RUN git config  --global --add advice.detachedHead false
RUN git config --global --add safe.directory /TASK && if [ $GIT_HASH != 'latest' ]; then git checkout $GIT_HASH;fi
RUN chown -R 1000:1000 /TASK
USER 1000:1000

