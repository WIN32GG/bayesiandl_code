cp -r ../../bayeformers bayeformers
cp ../../requirements.txt requirements.txt
DOCKER_BUILDKIT=1 docker build -t win32gg/squad_multitool --no-cache .
docker push win32gg/squad_multitool
rm -fr bayeformers
rm requirements.txt
