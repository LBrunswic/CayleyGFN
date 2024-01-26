FROM tensorflow:tf14
ARG GIT_REPO=CayleyGFN
ARG GIT_SERVER=10.0.0.85/git
ARG GIT_HASH=latest
RUN mkdir TASK
#RUN echo "10.0.0.85   node1" >> /etc/hosts
#RUN git clone http://$GIT_SERVER/$GIT_REPO TASK
WORKDIR /TASK
RUN mkdir LOGS
RUN mkdir RESULTS
RUN mkdir MODELS
RUN git config  --global --add advice.detachedHead false
RUN git config --global --add safe.directory /TASK && if [ $GIT_HASH != 'latest' ]; then git checkout $GIT_HASH;fi
#VOLUME /TASK/RESULTS
#VOLUME /TASK/LOGS
#VOLUME /TASK/MODELS
RUN chown -R 1000:1000 /TASK
#RUN chown -R 1000:1000 /TASK/LOGS
#RUN chown -R 1000:1000 /TASK/MODELS
#RUN chown -R 1000:1000 /TASK/RESULTS
USER 1000:1000
RUN cd Hyperparameters && python3 generate.py && python3 generate2.py && python3 generate_S15G3W1.py && python3 generate_S15G2W1.py
#CMD python3 docker_main.py
#ENTRYPOINT bash
