.PHONY: clean run

# Compiler
CC = nvcc

# Target
TARGET = $(basename $(SRC))

# Compile the CUDA file
$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC)

# run
run: $(TARGET)
	./$(TARGET)

# Clean
clean:
	rm -f $(TARGET)
