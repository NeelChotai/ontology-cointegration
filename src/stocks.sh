#!/bin/bash

curl -o nasdaq.txt ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt #ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt
cat nasdaq.txt | sed '1d;$d' | sed 's/|.*//g' | sed 'H;1h;$!d;x;y/\n/,/' | sed ':a;N;$!ba;s/\n//g' > stocks.txt
rm nasdaq.txt
