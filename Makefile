blur: main.cpp
	g++ -Wall main.cpp -fopenmp `pkg-config --libs opencv` -o blur
clean:
	rm -f mcf *.o

