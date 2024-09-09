<!-- to do list -->
- [ ] (ask amin?) to run the rate-distortion experiment with the symbolic bottleneck library.
- [ ] load vision transformer and audio model (wavenet)
- [ ] make the mnist dataset.
- [ ] find human audio data.

- The autoreg wrapper needs to change. to a more generic, simpler, sequential_output_wrapper and one_shot_output_wrapper.
the input to the wrapper is irrelevant. whether it is sequential or not.
This should make adding a decoder only model wrapper simple too. 

- I think we should get rid of the discrete-bottleneck showing up everywhere. it should be part of the backend library.
