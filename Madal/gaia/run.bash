#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <service_name>"
  exit 1
fi

SERVICE_NAME=$1
#!/bin/bash

PYTHON_SCRIPT="train.py"

services=(
  "dbservice1",
  "dbservice2",
  "logservice1",
  "logservice2",
  "mobservice1",
  "mobservice2",  
  "redisservice1",
  "redisservice2",
  "webservice1",
  "webservice2"
)

for service in "${services[@]}"
do
  echo "=============================="
  echo "Running training for: $service"
  echo "=============================="
  python "$PYTHON_SCRIPT" "$service"
  echo ""
done

python train.py "$SERVICE_NAME"
