---
sidebar_position: 2
---

# [ADVANCED] Serving Doodler as a web application for others to use

(page under construction)

These instructions assume you are using Ubuntu 20.04 (LTS) x64 on a Digital Ocean Droplet Virtual Machine (VM). You would need to adapt them to a different platform or OS

### Setting up the droplet
Below, you custom URL is called `$URL` and your IP address is written as `$IP` instead your actual IP address, XXX.XX.XXX.XXX

1. Under Networkin/Domains, add your `$URL` and www.`$URL` as separate domains, and redirect both of them to `$IP`
2. For each domain, you should have DNS records for 3 nameservers, ns1.digitalocean.com, ns2.digitalocean.com., and n3.digitalocean.com.
3. Those same nameservers should be linked as custom nameservers with your domain name provider (I used namecheap.com)

### Setting up newuser, firewalls, and installing anaconda

Below, the new user is called `newuser`, but you'd probably want to use something more suited to you, the sysadmin

```
ssh root@$IP
sudo adduser newuser
usermod -aG sudo newuser
ufw allow OpenSSH
ufw enable
exit
```

Log in as `newuser`:

```
ssh newuser@$IP
```

Install Anaconda3

```
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

Allow some ports and other tings on the firewall, ufw

```
sudo ufw allow 8050
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
```

### Downloading doodler code, setting up conda env

```
git clone --depth 1 https://github.com/dbuscombe-usgs/dash_doodler.git
cd dash_doodler
conda env create --file install/dashdoodler.yml
conda activate dashdoodler
conda install gunicorn
```

### Test locally

```
conda deactivate
```

Run this:

```
/home/newuser/anaconda3/envs/dashdoodler/bin/gunicorn --bind 0.0.0.0:8050 --timeout 600 doodler:server
```

And open a web browser and go here:

```
http://$IP:8050
```

You should see the Doodler app


### Setting up your web server, SSL and custom URL redirects

Install nginx (for web serving) and certbot (for ssl)

```
sudo apt install nginx
sudo apt-get install python3-certbot-nginx
```

Test getting your SSL certs in dry run mode

```
sudo certbot --nginx -d $URL -d www.$URL certonly --dry-run
```

Assuming that completed without error, run for real:

```
sudo certbot --nginx -d $URL -d www.$URL
```

See the certificate here: "https://www.ssllabs.com/ssltest/analyze.html?d=$URL&latest"


### Setting up Doodler as a background service

```
sudo nano /etc/systemd/system/doodler.service
```

```
[Unit]
Description=Gunicorn instance to serve myproject
After=network.target

[Service]
User=newuser
Group=www-data

WorkingDirectory=/home/newuser/dash_doodler
Environment="PATH=/home/newuser/anaconda3/envs/dashdoodler/bin"
ExecStart=/home/newuser/anaconda3/envs/dashdoodler/bin/gunicorn --workers 3 --bind unix:/home/newuser/dash_doodler/doodler.sock -m 007 --bind 0.0.0.0:8050 --timeout 600 doodler:server

[Install]
WantedBy=multi-user.target
```

Then run these commands

```
sudo systemctl stop doodler

sudo systemctl daemon-reload

sudo systemctl start doodler

sudo systemctl enable doodler

sudo systemctl status doodler
```

open a web browser and go here:

```
http://$IP:8050
```

And you should see Doodler. The service works. This should work on reboot, and in the background. Check on it any time using

```
sudo systemctl status doodler
```

### Setting $URL redirect

However, I couldn't get doodler to appear at my `$URL` unless I also edited this file:

```
sudo nano /etc/nginx/sites-available/default
```

Like this (replacing `$URL` with whatever it is):

```
        # Add index.php to the list if you are using PHP
        index index.html index.htm index.nginx-debian.html;

        server_name $URL www.$URL;

        location / {
                # First attempt to serve request as file, then
                # as directory, then fall back to displaying a 404.
                #try_files $uri $uri/ =404;
    location / {
        proxy_pass http://0.0.0.0:8050; #whatever port your app runs on
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
        }
```

```
sudo nano /etc/nginx/sites-available/doodler
```

Should look like this (replacing `$URL` with whatever it is):

```
server {
    server_name  $URL www. $URL;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/newuser/doodler/doodler.sock;
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/ $URL/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/ $URL/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot


}

server {
    if ($host =  $URL) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    if ($host =  $URL) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    listen 80;
    server_name $URL www.$URL;
    return 404; # managed by Certbot


}

```

Then symlink

```
sudo ln -s /etc/nginx/sites-available/doodler /etc/nginx/sites-enabled
```

Test the config files
```
sudo nginx -t
```

I got "nginx: [warn] conflicting server name" warnings because of the two nginx enabled sites pointing to the same place

Restart nginx and update the firewall

```
sudo systemctl restart nginx
sudo ufw allow 'Nginx Full'
```

ow your doodler service will be running the app at

```
http://$IP:8050
```

which redirects to

```
http://$URL
```

But, despite the SSL cert from certbot, for some unknown reason it does not appear at

```
https://$URL
```

as expected

any thoughts? Get in touch!


I carried out this process and have a site hosted [here](http://doodler.xyz/)
