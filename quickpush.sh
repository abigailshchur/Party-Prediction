cd ./
t=$(date +"%H:%M:%S")
d=$(date +"%m/%d/%Y")
git add .
git commit -m "pushed code at $t on $d"
git push