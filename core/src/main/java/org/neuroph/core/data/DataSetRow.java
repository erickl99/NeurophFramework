/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.core.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import org.neuroph.util.VectorParser;

/**
 * This class represents a single data row in a data set. It has input and
 * desired output for supervised learning rules. It can also be used only with
 * input for unsupervised learning rules.
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class DataSetRow implements Serializable {

    /**
     * The class fingerprint that is set to indicate serialization compatibility
     * with a previous version of the class
     */
    private static final long serialVersionUID = 1L;
    /**
     * Input vector for this row.
     */
    protected double[] input;

    /**
     * Desired output used for supervised learning algorithms.
     */
    private double[] desiredOutput;

    /**
     * Creates new training element with specified input and desired output
     * vectors specifed as strings
     *
     * @param input input vector as space separated string
     * @param desiredOutput desired output vector as space separated string
     */
    public DataSetRow(String input, String desiredOutput) {
        this.input = VectorParser.parseDoubleArray(input);
        this.desiredOutput = VectorParser.parseDoubleArray(desiredOutput);
    }

    /**
     * Creates new training element with specified input and desired output
     * vectors
     *
     * @param input input array
     * @param desiredOutput desired output array
     */
    public DataSetRow(double[] input, double[] desiredOutput) {
        this.input = input;
        this.desiredOutput = desiredOutput;
    }

    /**
     * Creates new training element with input array
     *
     * @param input input array
     */
    public DataSetRow(double... input) {
        this.input = input;
    }

    /**
     * Creates new training element with specified input and desired output
     * vectors
     *
     * @param input input vector
     * @param desiredOutput desired output vector
     */
    public DataSetRow(ArrayList<Double> input,
            ArrayList<Double> desiredOutput) {
        this.input = VectorParser.toDoubleArray(input);
        this.desiredOutput = VectorParser.toDoubleArray(desiredOutput);
    }

    public DataSetRow(ArrayList<Double> input) {
        this.input = VectorParser.toDoubleArray(input);
    }

    /**
     * Returns input vector
     *
     * @return input vector
     */
    public double[] getInput() {
        return this.input;
    }

    /**
     * Sets input vector
     *
     * @param input input vector
     */
    public void setInput(double[] input) {
        this.input = input;
    }

    public double[] getDesiredOutput() {
        return desiredOutput;
    }

    public void setDesiredOutput(double[] desiredOutput) {
        this.desiredOutput = desiredOutput;
    }

    public boolean isSupervised() {
        return (desiredOutput != null);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Input: ");
        for (double in : input) {
            sb.append(in).append(", ");
        }
        sb.delete(sb.length() - 2, sb.length() - 1);

        if (isSupervised()) {
            sb.append(" Desired output: ");
            for (double out : desiredOutput) {
                sb.append(out).append(", ");
            }
            sb.delete(sb.length() - 2, sb.length() - 1);
        }

        return sb.toString();
    }

    public double[] toArray() {
        double[] row = new double[input.length + desiredOutput.length];
        System.arraycopy(input, 0, row, 0, input.length);
        System.arraycopy(desiredOutput, 0, row, input.length, desiredOutput.length);
        return row;
    }

    public String toCSV() {
        StringBuilder sb = new StringBuilder();

        for (double in : input) {
            sb.append(in).append(", ");
        }

        if (isSupervised()) {
            for (double out : desiredOutput) {
                sb.append(out).append(", ");
            }
        }

        sb.delete(sb.length() - 2, sb.length() - 1);

        return sb.toString();
    }

    @Override
    public int hashCode() {
        int hash = 5;
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final DataSetRow other = (DataSetRow) obj;
        if (!Arrays.equals(this.input, other.input)) {
            return false;
        }
        if (!Arrays.equals(this.desiredOutput, other.desiredOutput)) {
            return false;
        }
        return true;
    }

}
