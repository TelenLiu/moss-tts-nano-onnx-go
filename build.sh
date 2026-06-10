dot="."
year=`date +%Y`
year=${year:2:2}
mm=`date +%m`
if [ ${mm:0:1} == "0" ]
then
    mm=${mm:1:1}
fi
dd=`date +%d`
if [ ${dd:0:1} == "0" ]
then
    dd=${dd:1:1}
fi
version=$year$dot$mm$dot$dd`date +%H`
go build -ldflags="-X main.Version=$version" -o moss-tts-nano-onnx-go ./cmd/