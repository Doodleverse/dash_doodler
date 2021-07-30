# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020-2021, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM continuumio/miniconda3
LABEL maintainer "Doodler, by Dr Daniel Buscombe, Marda Science/USGS <dbuscombe@contractor.usgs.gov>"
WORKDIR /
# The code to run when container is started:
COPY ./ ./

COPY environment/dashdoodler-clean.yml .
RUN conda env create -f dashdoodler-clean.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "dashdoodler", "/bin/bash", "-c"]

EXPOSE 8050/tcp
EXPOSE 8050/udp
EXPOSE 80
EXPOSE 8080

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dashdoodler", "python", "doodler.py"]
