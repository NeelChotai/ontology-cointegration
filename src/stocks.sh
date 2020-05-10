#!/bin/bash

curl -o nasdaq.txt ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt
cat nasdaq.txt | grep -Eo '^\w\|\w*' | sed 's/^\w|//g' | sed 'H;1h;$!d;x;y/\n/,/' | sed ':a;N;$!ba;s/\n//g' > stocks.txt
rm nasdaq.txt