FROM tensorflow:tf14
#
WORKDIR /tf
ARG GIT_REPO=CayleyGFN
ARG GIT_SERVER=node1/git
#ARG GIT_SERVER_LOCATION=/home/maxbrain/AI/
ARG GIT_HASH=74487dc7da31d14c4b4353bfd5383513a1be8e57
RUN cd /tf
#&& ping 192.168.50.167
RUN mkdir TASK
RUN git clone http://$GIT_SERVER/$GIT_REPO TASK
RUN chown -R 1000:1000 TASK
RUN cd TASK
RUN git checkout $GIT_HASH
VOLUME /tf/TASK/RESULTS
VOLUME /tf/TASK/LOG

USER 1000:1000
CMD bash && cd TASK && python3
#ARG RESULTS_DIR=/home/maxbraiexitn/IA/results/CayleyGFN