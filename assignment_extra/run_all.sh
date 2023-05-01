for FILE in config/*;
do
    python3 main.py --config $FILE;
done