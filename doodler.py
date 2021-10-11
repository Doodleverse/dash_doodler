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

#========================================================
## ``````````````````````````` imports
##========================================================
from app import app, server

from environment.settings import APP_HOST, APP_PORT, APP_DEBUG, DEV_TOOLS_PROPS_CHECK, APP_DOWNLOAD_SAMPLE, APP_THREADED
import os, zipfile, requests
from datetime import datetime
from glob import glob

if __name__ == "__main__":

    #========================================================
    ## downlaod the sample if APP_DOWNLOAD_SAMPLE is True
    #========================================================
    if APP_DOWNLOAD_SAMPLE:
        url='https://github.com/dbuscombe-usgs/dash_doodler/releases/download/data/sample_images.zip'
        filename = os.path.join(os.getcwd(), "sample_images.zip")
        r = requests.get(url, allow_redirects=True)
        open(filename, 'wb').write(r.content)

        with zipfile.ZipFile(filename, "r") as z_fp:
            z_fp.extractall("./assets/")
        os.remove(filename)

    #if labeled images exist in labaled folder, zip them up with a timestamp, and remove the individual files
    try:
        filename = 'labeled'+os.sep+'labeled-'+datetime.now().strftime("%Y-%m-%d-%H-%M")+'.zip'
        with zipfile.ZipFile(filename, "w") as z_fp:
            for k in glob("./labeled/*.jpeg")+glob("./labeled/*.JPG")+glob("./labeled/*.jpg"):
                z_fp.write(k)
        z_fp.close()

        # for k in glob("./labeled/*.jpeg")+glob("./labeled/*.JPG")+glob("./labeled/*.jpg"):
        #     os.remove(k)

    except:
        pass

    #========================================================
    ## ``````````````````````````` run the app in the browser at $APP_HOST, port $APP_PORT
    ##========================================================
    print('Go to http://%s:%i/ in your web browser to use Doodler'% (APP_HOST,APP_PORT))
    app.run_server(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        dev_tools_props_check=DEV_TOOLS_PROPS_CHECK,
        threaded=APP_THREADED
    )
