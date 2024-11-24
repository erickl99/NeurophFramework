/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.neuroph.nnet.learning;

import java.io.Serializable;
import org.neuroph.core.Connection;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.learning.SupervisedLearning;

/**
 * LMS learning rule for neural networks. This learning rule is used to train
 * Adaline neural network, and this class is base for all LMS based learning
 * rules like PerceptronLearning, DeltaRule, SigmoidDeltaRule, Backpropagation
 * etc.
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class LMS extends SupervisedLearning implements Serializable {

    /**
     * The class fingerprint that is set to indicate serialization
     * compatibility with a previous version of the class.
     */
    private static final long serialVersionUID = 2L;


    /**
     * Creates a new LMS learning rule
     */
    public LMS() {

    }


    /**
     * This method calculates weight change for the network's output neurons for the given output error vector.
     *
     * @param outputError
     *            output error vector for some network input- the difference between desired and actual output
     * @see SupervisedLearning#learnPattern(org.neuroph.core.data.DataSetRow)  learnPattern
     */
    @Override
    protected void calculateWeightChanges(final double[] outputError) {
        int i = 0;
        for (Neuron neuron : neuralNetwork.getOutputNeurons()) {
            neuron.setDelta(outputError[i]);    // set the neuron error/delta, as difference between desired and actual output
            calculateWeightChanges(neuron);     // and calculate weight changes
            i++;
        }
    }

    /**
     * This method calculates weights changes for the single neuron.
     * It iterates through all neuron's input connections, and calculates/set weight change for each weight
     * using formula
     *      deltaWeight = -learningRate * delta * input
     *
     * where delta is a neuron error, a difference between desired/target and actual output for specific neuron
     *      neuronError = desiredOutput[i] - actualOutput[i] (see method SuprevisedLearning.calculateOutputError)
     *
     * @param neuron
     *            neuron to update weights
     *
     * @see LMS#calculateWeightChanges(double[])
     */
    protected void calculateWeightChanges(Neuron neuron) {
        // get the error(delta) for the specified neuron,
        double delta = neuron.getDelta();

        // tanh can be used to minimise the impact of big error values, which can cause network instability
        // suggested at https://sourceforge.net/tracker/?func=detail&atid=1107579&aid=3130561&group_id=238532
        // double neuronError = Math.tanh(neuron.getError());

        // iterate through all neuron's input connections
        for (Connection connection : neuron.getInputConnections()) {
            // get the input from current connection
            final double input = connection.getInput();
            // calculate the weight change
            final double weightChange = -learningRate * delta * input;

            // get the connection weight
            final Weight weight = connection.getWeight();

            // if the learning is in online mode (not batch) apply the weight change immediately
            if (!this.isBatchMode()) {
                weight.weightChange = weightChange;
            } else { // otherwise if its in batch mode, accumulate  weight changes and apply them after the current epoch (see SupervisedLearning.doLearningEpoch method)
                weight.weightChange += weightChange;
            }
        }
    }

}
