gcloud ml-engine local train \
  --module-name trainer.example5 \
  --package-path ./trainer \
  -- \
  --train-file sentiment_set.pickle \
  --job-dir ./tmp/example-5