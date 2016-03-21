echo "Type a commmit message, followed by [ENTER]:"
read msg

cd "/Volumes/Macintosh_Extension/Documents_Extension/Google Drive/Cornell/Semester 6/CS 4300/Assignments/Final-Project/cs4300sp2016-party-prediction"
t=$(date +"%H:%M:%S")
d=$(date +"%m/%d/%Y")
git add .
git commit -m "$msg [pushed at $t on $d]"
git push