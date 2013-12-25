package org.neuroph.netbeans.main.easyneurons.dialog;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.logging.Logger;
import org.netbeans.api.settings.ConvertAsProperties;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.netbeans.main.LearningInfo;
import org.neuroph.netbeans.visual.NeuralNetAndDataSet;
import org.neuroph.nnet.learning.LMS;
import org.openide.util.NbBundle;
import org.openide.windows.TopComponent;
import org.openide.windows.WindowManager;

/**
 * Top component which displays info about running supervised learning rule.
 */
@ConvertAsProperties(dtd = "-//org.neuroph.netbeans.main.easyneurons.dialog//SupervisedTrainingMonitor//EN",
autostore = false)
public final class SupervisedTrainingMonitorTopComponent extends TopComponent implements LearningEventListener { // not observer but listener!

    private static SupervisedTrainingMonitorTopComponent instance;
    /** path to the icon used by the component and its open action */
//    static final String ICON_PATH = "SET/PATH/TO/ICON/HERE";
    private static final String PREFERRED_ID = "SupervisedTrainingMonitorTopComponent";

    public SupervisedTrainingMonitorTopComponent() {
        initComponents();
        setName(NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "CTL_SupervisedTrainingMonitorTopComponent"));
        setToolTipText(NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "HINT_SupervisedTrainingMonitorTopComponent"));
//        setIcon(ImageUtilities.loadImage(ICON_PATH, true));

    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        totalNetErrorField = new javax.swing.JTextField();
        errLabel = new javax.swing.JLabel();
        iterationLabel = new javax.swing.JLabel();
        currentIterationField = new javax.swing.JTextField();
        stopButton = new javax.swing.JButton();
        pauseButton = new javax.swing.JButton();

        totalNetErrorField.setColumns(18);

        org.openide.awt.Mnemonics.setLocalizedText(errLabel, org.openide.util.NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "SupervisedTrainingMonitorTopComponent.errLabel.text")); // NOI18N

        org.openide.awt.Mnemonics.setLocalizedText(iterationLabel, org.openide.util.NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "SupervisedTrainingMonitorTopComponent.iterationLabel.text")); // NOI18N

        currentIterationField.setColumns(10);

        org.openide.awt.Mnemonics.setLocalizedText(stopButton, org.openide.util.NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "SupervisedTrainingMonitorTopComponent.stopButton.text")); // NOI18N
        stopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                stopButtonActionPerformed(evt);
            }
        });

        org.openide.awt.Mnemonics.setLocalizedText(pauseButton, org.openide.util.NbBundle.getMessage(SupervisedTrainingMonitorTopComponent.class, "SupervisedTrainingMonitorTopComponent.pauseButton.text")); // NOI18N
        pauseButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                pauseButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                .addComponent(errLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(totalNetErrorField, javax.swing.GroupLayout.PREFERRED_SIZE, 1, Short.MAX_VALUE))
                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                                .addComponent(iterationLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(currentIterationField, javax.swing.GroupLayout.DEFAULT_SIZE, 99, Short.MAX_VALUE))))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(49, 49, 49)
                        .addComponent(pauseButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(stopButton)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(errLabel)
                    .addComponent(totalNetErrorField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(iterationLabel)
                    .addComponent(currentIterationField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(pauseButton)
                    .addComponent(stopButton))
                .addContainerGap(197, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void stopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_stopButtonActionPerformed
        stop();
    }//GEN-LAST:event_stopButtonActionPerformed

    private void pauseButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_pauseButtonActionPerformed
        pause();
    }//GEN-LAST:event_pauseButtonActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField currentIterationField;
    private javax.swing.JLabel errLabel;
    private javax.swing.JLabel iterationLabel;
    private javax.swing.JButton pauseButton;
    private javax.swing.JButton stopButton;
    private javax.swing.JTextField totalNetErrorField;
    // End of variables declaration//GEN-END:variables
    /**
     * Gets default instance. Do not use directly: reserved for *.settings files only,
     * i.e. deserialization routines; otherwise you could get a non-deserialized instance.
     * To obtain the singleton instance, use {@link #findInstance}.
     */
    public static synchronized SupervisedTrainingMonitorTopComponent getDefault() {
        if (instance == null) {
            instance = new SupervisedTrainingMonitorTopComponent();
        }
        return instance;
    }

    /**
     * Obtain the SupervisedTrainingMonitorFrameTopComponent instance. Never call {@link #getDefault} directly!
     */
    public static synchronized SupervisedTrainingMonitorTopComponent findInstance() {
        TopComponent win = WindowManager.getDefault().findTopComponent(PREFERRED_ID);
        if (win == null) {
            Logger.getLogger(SupervisedTrainingMonitorTopComponent.class.getName()).warning(
                    "Cannot find " + PREFERRED_ID + " component. It will not be located properly in the window system.");
            return getDefault();
        }
        if (win instanceof SupervisedTrainingMonitorTopComponent) {
            return (SupervisedTrainingMonitorTopComponent) win;
        }
        Logger.getLogger(SupervisedTrainingMonitorTopComponent.class.getName()).warning(
                "There seem to be multiple components with the '" + PREFERRED_ID
                + "' ID. That is a potential source of errors and unexpected behavior.");
        return getDefault();
    }

    @Override
    public int getPersistenceType() {
        return TopComponent.PERSISTENCE_ALWAYS;
    }

    @Override
    public void componentOpened() {
        // TODO add custom code on component opening
    }

    @Override
    public void componentClosed() {
        this.stop();
        try { // ugly hack to catch NullPointerException which happens sometimes when closing this window
            this.trainingController.getNetwork().getLearningRule().removeListener(this) ; // deletListeners removeListeners
        } catch(Exception npe) {
            npe.printStackTrace();
        }
    }

    void writeProperties(java.util.Properties p) {
        // better to version settings since initial version as advocated at
        // http://wiki.apidesign.org/wiki/PropertyFiles
        p.setProperty("version", "1.0");
        // TODO store your settings
    }

    Object readProperties(java.util.Properties p) {
        if (instance == null) {
            instance = this;
        }
        instance.readPropertiesImpl(p);
        return instance;
    }

    private void readPropertiesImpl(java.util.Properties p) {
        String version = p.getProperty("version");
        // TODO read your settings according to their version
    }

    @Override
    protected String preferredID() {
        return PREFERRED_ID;
    }

    NeuralNetAndDataSet trainingController;
    boolean userPaused = false;
    ConcurrentLinkedQueue<LearningInfo> dataQueueBuffer;
    GuiWorker guiWorker;

    public void setSupervisedTrainingMonitorFrameVariables(NeuralNetAndDataSet controller) {
        this.trainingController = controller;
        initComponents();
        dataQueueBuffer = new ConcurrentLinkedQueue<LearningInfo>();
        guiWorker = new GuiWorker();
        guiWorker.start();
    }

        public void observe(LearningRule learningRule) {
             learningRule.addListener(this);
        }


    public void stop() {
        if(this.trainingController!=null) {
            this.trainingController.stopTraining();
        }
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
         LMS learningRule = (LMS) le.getSource(); // put this in event handler

                // get data that we want to display
                final LearningInfo learningInfo = new LearningInfo( learningRule.getCurrentIteration(),
                                                                    learningRule.getTotalNetworkError());
                guiWorker.putData(learningInfo);                

                if (learningRule.isStopped()) {
                    guiWorker.setFinnished(true);
                    guiWorker = null;
                    learningRule.removeListener(this);
                }
    }
    
    private class GuiWorker extends Thread {
        boolean finnished = false;
        boolean hasData = false;
        
        synchronized public void  setHasData(boolean hasData) {
            this.hasData = hasData;
        }

        
        @Override
        public void run() {
                while (!finnished) {
                   //Wait until data is available.
                       synchronized(this) {
                            while (!hasData) {
                                try {
                                    wait();
                                } catch (InterruptedException e) {}
                            }
                       }

                     setHasData(false);   
                       
                    while(!dataQueueBuffer.isEmpty()) {   
                        LearningInfo li = dataQueueBuffer.poll();
                        currentIterationField.setText(li.getIteration().toString());
                        totalNetErrorField.setText(li.getError().toString());
                    }          
                    

                }           
        }
        
        public void setFinnished(boolean finnished) {
            this.finnished = finnished;
        }      
        
        
        private void putData(LearningInfo learningInfo) {
            dataQueueBuffer.add(learningInfo);
            setHasData(true);

            synchronized(this) {
                  this.notify(); 
            }
          
        }        
        
    }

    public void pause() {
       if (!userPaused) {
           userPaused = true;
           trainingController.pause();
           pauseButton.setText("Resume");
           stopButton.setEnabled(false);
       } else {
           trainingController.resume();
           userPaused = false;
           pauseButton.setText("Pause");
           stopButton.setEnabled(true);
       }
    }


}