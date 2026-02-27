CSOURCE = main.c
COBJECT = main
CFLAGS = -lm
DATAFILE = Fitness.data

Fitness.png: FitnessPlot.gp Fitness.data
	gnuplot $<

$(DATAFILE): $(COBJECT)
	./$(COBJECT)

$(COBJECT): $(CSOURCE)
	$(CC) $(CSOURCE) -o $(COBJECT) $(CFLAGS) 

clean:
	rm -f *.png *.data $(COBJECT)