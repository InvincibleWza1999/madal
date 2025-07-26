#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <service_name>"
  exit 1
fi

SERVICE_NAME=$1
#!/bin/bash

PYTHON_SCRIPT="train.py"

services=(
  "paymentservice-9cdb6588f-554sm"
  "checkoutservice-578fcf4766-9csqn"
  "recommendationservice-6cfdd55578-gfj6q"
  "currencyservice-cf787dd48-vpjrd"
  "shippingservice-7b598fc7d-lmggd"
  "cartservice-579f59597d-wc2lz"
  "frontend-579b9bff58-t2dbm"
  "adservice-5f6585d649-fnmft"
  "emailservice-55fdc5b988-f6xth"
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
