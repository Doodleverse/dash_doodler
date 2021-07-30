HOST="127.0.0.1" #"0.0.0.0"  #for deployment
PORT="8050"
DEBUG=False
DEV_TOOLS_PROPS_CHECK=False
DOWNLOAD_SAMPLE=False #True

## uncomment line below to prevent the program from downloading the sample imagery into the assets folder
#DOWNLOAD_SAMPLE=False

APP_HOST = str(HOST)
APP_PORT = int(PORT)
APP_DEBUG = bool(DEBUG)
DEV_TOOLS_PROPS_CHECK = bool(DEV_TOOLS_PROPS_CHECK)
APP_DOWNLOAD_SAMPLE = bool(DOWNLOAD_SAMPLE)
