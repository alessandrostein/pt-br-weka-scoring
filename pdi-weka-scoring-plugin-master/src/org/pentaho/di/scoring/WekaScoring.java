/*
* This program is free software; you can redistribute it and/or modify it under the
* terms of the GNU General Public License, version 2 as published by the Free Software
* Foundation.
*
* You should have received a copy of the GNU General Public License along with this
* program; if not, you can obtain a copy at http://www.gnu.org/licenses/gpl-2.0.html
* or from the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
* This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
*
* Copyright 2006 - 2013 Pentaho Corporation.  All rights reserved.
*/

package org.pentaho.di.scoring;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.pentaho.di.core.Const;
import org.pentaho.di.core.exception.KettleException;
import org.pentaho.di.core.row.RowMetaInterface;
import org.pentaho.di.i18n.BaseMessages;
import org.pentaho.di.trans.Trans;
import org.pentaho.di.trans.TransMeta;
import org.pentaho.di.trans.step.BaseStep;
import org.pentaho.di.trans.step.StepDataInterface;
import org.pentaho.di.trans.step.StepInterface;
import org.pentaho.di.trans.step.StepMeta;
import org.pentaho.di.trans.step.StepMetaInterface;

import weka.core.BatchPredictor;
import weka.core.Instances;
import weka.core.SerializedObject;

/**
 * Applies a pre-built weka model (classifier or clusterer) to incoming rows and
 * appends predictions. Predictions can be a label (classification/clustering),
 * a number (regression), or a probability distribution over classes/clusters.
 * <p>
 * 
 * Attributes that the Weka model was constructed from are automatically mapped
 * to incoming Kettle fields on the basis of name and type. Any attributes that
 * cannot be mapped due to type mismatch or not being present in the incoming
 * fields receive missing values when incoming Kettle rows are converted to
 * Weka's Instance format. Similarly, any values for string fields that have not
 * been seen during the training of the Weka model are converted to missing
 * values.
 * 
 * PT-BR
 * 
 * Aplica um pre-built do modelo weka (classificador ou agrupador) das linhas de 
 * entradas e anexa previsoes. Previsoes podem ser uma label (classificacao, agrupamento),
 * um number (Regressao), ou uma probabilidade de distribuicao sobre classes/agrupadores.
 * 
 * Atributos que o modelo foi construido atraves do Weka sao mapeados automaticamente
 * para os campos Kettle de entrada na base do nome e tipo. Algum atributo que nao pode
 * ser mapeado devido a incompatibilidade do tipo ou nao estar presente nos campos de entrada
 * que nao tenham valores onde as entradas do Kettle sao convertidadas para o formato 
 * Weka's Instance. De forma similar, alguns valores para campos Strings que nao tenham 
 * sido visto durante o treinamento do modelo Weka sao convertidos para valores desconhecidos.
 * 
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}org)
 */
public class WekaScoring extends BaseStep implements StepInterface {

  private WekaScoringMeta m_meta;
  private WekaScoringData m_data;

  /** only used when grabbing model file names from the incoming stream
   * 
   * PT-BR
   * 
   * Utilizado quando capturar os nomes dos arquivos do fluxo de entrada.
   */
  private int m_indexOfFieldToLoadFrom = -1;

  /**
   * cache for models that are loaded from files specified in incoming rows
   * 
   * PT-BR
   * 
   * Armazena para modelos que sao carregados dos arquivos especificos nas 
   * linhas de entrada.
   */
  private Map<String, WekaScoringModel> m_modelCache;

  /**
   * model filename from the last row processed (if reading model filenames from
   * a row field
   * 
   * PT-BR
   * 
   * Modelo do nome do arquivo da ultima linha processada (se a leitura do
   * modelo do nome do arquivo para um campo da linha).  
   */
  private String m_lastRowModelFile = ""; //$NON-NLS-1$

  /** size of the batches of rows to be scored if the model is a batch scorer 
   * 
   * PT-BR
   * 
   * Tamanho dos lotes das linhas para ser marcado se o modelo e um lote marcador.
   * 
   */
  private int m_batchScoringSize = WekaScoringMeta.DEFAULT_BATCH_SCORING_SIZE;
  private List<Object[]> m_batch;

  /**
   * Creates a new <code>WekaScoring</code> instance
   * 
   * PT-BR
   * 
   * Criar uma nova instancia WekaScoring.
   * 
   * @param stepMeta holds the step's meta data
   *                 Contem os dados do StepMeta
   * @param stepDataInterface holds the step's temporary data
   *                          Contem os dados temporarios do StepMeta  
   * @param copyNr the number assigned to the step
   *               O numero atribuido para o Step.   
   * @param transMeta meta data for the transformation
   *                  meta data para a transformacao  
   * @param trans a <code>Trans</code> value
   *              um valor Trans  
   */
  public WekaScoring(StepMeta stepMeta, StepDataInterface stepDataInterface,
      int copyNr, TransMeta transMeta, Trans trans) {
    super(stepMeta, stepDataInterface, copyNr, transMeta, trans);
  }

  /**
   * Sets the model to use from the path supplied in a user chosen field of the
   * incoming data stream. User may opt to have loaded models cached in memory.
   * User may also opt to supply a default model to be used when there is none
   * specified in the field value.
   * 
   * PT-BR
   * 
   * Seta o modelo para usar a partir do campo do caminho fornecido por um usuario  
   * atraves do fluxo de dados de entrada. Usuario pode optar por ter modelos carregados
   * em cache de memoria. Usuario pode tambem optar por  fornecer um modelo padrao 
   * para usar quando nao encontrar nenhum campo com valor especifico.
   * 
   * @param row
   * @throws KettleException
   */
  private void setModelFromField(Object[] row) throws KettleException {

    RowMetaInterface inputRowMeta = getInputRowMeta();
    String modelFileName = inputRowMeta
        .getString(row, m_indexOfFieldToLoadFrom);

    if (Const.isEmpty(modelFileName)) {
      // see if there is a default model to use
      // Verifique se possui um modelo padrao a ser usado.
      WekaScoringModel defaultM = m_data.getDefaultModel();
      if (defaultM == null) {
        throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
            "WekaScoring.Error.NoModelFileSpecifiedInFieldAndNoDefaultModel")); //$NON-NLS-1$
      }
      logDebug(BaseMessages.getString(WekaScoringMeta.PKG,
          "WekaScoring.Debug.UsingDefaultModel")); //$NON-NLS-1$
      m_data.setModel(defaultM);
      return;
    }

    String resolvedName = environmentSubstitute(modelFileName);

    if (resolvedName.equals(m_lastRowModelFile)) {
      // nothing to do, just return
      // Nada para fazer, apenas retorne.
      return;
    }

    if (m_meta.getCacheLoadedModels()) {
      WekaScoringModel modelToUse = m_modelCache.get(resolvedName);
      if (modelToUse != null) {
        logDebug(BaseMessages.getString(WekaScoringMeta.PKG,
            "WekaScoring.Debug.FoundModelInCache") //$NON-NLS-1$
            + " " //$NON-NLS-1$
            + modelToUse.getModel().getClass());
        m_data.setModel(modelToUse);
        m_lastRowModelFile = resolvedName;
        return;
      }
    }

    // load the model
    // Carrega o modelo
    logDebug(BaseMessages.getString(WekaScoringMeta.PKG,
        "WekaScoring.Debug.LoadingModelUsingFieldValue") //$NON-NLS-1$
        + " " //$NON-NLS-1$
        + environmentSubstitute(modelFileName));
    WekaScoringModel modelToUse = setModel(modelFileName);

    if (m_meta.getCacheLoadedModels()) {
      m_modelCache.put(resolvedName, modelToUse);
    }
  }

  private WekaScoringModel setModel(String modelFileName)
      throws KettleException {

    /*
     * String modName = environmentSubstitute(modelFileName); File modelFile =
     * null; if (modName.startsWith("file:")) { //$NON-NLS-1$ try { modName =
     * modName.replace(" ", "%20"); //$NON-NLS-1$ //$NON-NLS-2$ modelFile = new
     * File(new java.net.URI(modName)); } catch (Exception ex) { throw new
     * KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
     * "WekaScoring.Error.MalformedURIForModelFile"), ex); //$NON-NLS-1$ } }
     * else { modelFile = new File(modName); } if (!modelFile.exists()) { throw
     * new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
     * "WekaScoring.Error.NonExistentModelFile", modName)); //$NON-NLS-1$ }
     */

    // Load the model
    // Carrega o modelo
    WekaScoringModel model = null;
    try {
      model = WekaScoringData.loadSerializedModel(modelFileName,
          getLogChannel(), this);
      m_data.setModel(model);

      if (m_meta.getFileNameFromField()) {
        m_lastRowModelFile = environmentSubstitute(modelFileName);
      }
    } catch (Exception ex) {
      throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
          "WekaScoring.Error.ProblemDeserializingModel"), ex); //$NON-NLS-1$
    }
    return model;
  }

  /**
   * Process an incoming row of data.
   * 
   * PT-BR
   * 
   * Processar uma linha de entradada de dados  
   * 
   * @param smi a <code>StepMetaInterface</code> value
   * @param sdi a <code>StepDataInterface</code> value
   * @return a <code>boolean</code> value
   * @exception KettleException if an error occurs
   */
  @Override
  public boolean processRow(StepMetaInterface smi, StepDataInterface sdi)
      throws KettleException {

    m_meta = (WekaScoringMeta) smi;
    m_data = (WekaScoringData) sdi;

    Object[] r = getRow();

    if (r == null) {
      if (m_data.getModel().isBatchPredictor()
          && !m_meta.getFileNameFromField() && m_batch.size() > 0) {
        try {
          outputBatchRows();
        } catch (Exception ex) {
          throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
              "WekaScoring.Error.ProblemWhileGettingPredictionsForBatch"), ex); //$NON-NLS-1$
        }
      }

      // see if we have an incremental model that is to be saved somewhere.
      // Verifique se tenha um modelo incrementar que possa estar salvo em algum lugar. 
      if (!m_meta.getFileNameFromField() && m_meta.getUpdateIncrementalModel()) {
        if (!Const.isEmpty(m_meta.getSavedModelFileName())) {
          // try and save that sucker...
          // Testar e Salvar que sucker ...
          try {
            String modName = environmentSubstitute(m_meta
                .getSavedModelFileName());
            File updatedModelFile = null;
            if (modName.startsWith("file:")) { //$NON-NLS-1$
              try {
                modName = modName.replace(" ", "%20"); //$NON-NLS-1$ //$NON-NLS-2$
                updatedModelFile = new File(new java.net.URI(modName));
              } catch (Exception ex) {
                throw new KettleException(BaseMessages.getString(
                    WekaScoringMeta.PKG,
                    "WekaScoring.Error.MalformedURIForUpdatedModelFile"), ex); //$NON-NLS-1$
              }
            } else {
              updatedModelFile = new File(modName);
            }
            WekaScoringData.saveSerializedModel(m_data.getModel(),
                updatedModelFile);
          } catch (Exception ex) {
            throw new KettleException(BaseMessages.getString(
                WekaScoringMeta.PKG,
                "WekaScoring.Error.ProblemSavingUpdatedModelToFile"), ex); //$NON-NLS-1$
          }
        }
      }

      if (m_meta.getFileNameFromField()) {
        // clear the main model
        // Limpar o modelo principal
        m_data.setModel(null);
      } else {
        m_data.getModel().done();
      }

      setOutputDone();
      return false;
    }

    // Handle the first row
    // Manipula a primeira linha.
    if (first) {
      first = false;

      m_data.setOutputRowMeta(getInputRowMeta().clone());
      if (m_meta.getFileNameFromField()) {
        RowMetaInterface inputRowMeta = getInputRowMeta();

        m_indexOfFieldToLoadFrom = inputRowMeta.indexOfValue(m_meta
            .getFieldNameToLoadModelFrom());

        if (m_indexOfFieldToLoadFrom < 0) {
          throw new KettleException("Unable to locate model file field " //$NON-NLS-1$
              + m_meta.getFieldNameToLoadModelFrom()
              + " in the incoming stream!"); //$NON-NLS-1$
        }

        if (!inputRowMeta.getValueMeta(m_indexOfFieldToLoadFrom).isString()) {
          throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
              "WekaScoring.Error.")); //$NON-NLS-1$
        }

        if (m_meta.getCacheLoadedModels()) {
          m_modelCache = new HashMap<String, WekaScoringModel>();
        }

        // set the default model
        // Seta o modelo padrao
        if (!Const.isEmpty(m_meta.getSerializedModelFileName())) {
          WekaScoringModel defaultModel = setModel(m_meta
              .getSerializedModelFileName());

          m_data.setDefaultModel(defaultModel);
        } else if (m_meta.getModel() != null) {
          try {
            SerializedObject so = new SerializedObject(m_meta.getModel());
            WekaScoringModel defaultModel = (WekaScoringModel) so.getObject();

            m_data.setDefaultModel(defaultModel);
          } catch (Exception ex) {
            throw new KettleException(ex);
          }
        }

        // set the main model from this row
        // Seta o modelo principal para esta linha
        setModelFromField(r);
        logBasic(BaseMessages.getString(WekaScoringMeta.PKG,
            "WekaScoring.Message.SourcingModelNamesFromInputField", //$NON-NLS-1$
            m_meta.getFieldNameToLoadModelFrom()));
      } else if (m_meta.getModel() == null
          || !Const.isEmpty(m_meta.getSerializedModelFileName())) {
        // If we don't have a model, or a file name is set, then load from file
        // Se nao tem um modelo, ou o nome do arquivo esta setado, depos carrega o arquivo.

        // Check that we have a file to try and load a classifier from
        // Verificar  que tenha um arquivo para testar e carregar uma classificao
        if (Const.isEmpty(m_meta.getSerializedModelFileName())) {
          throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
              "WekaScoring.Error.NoFilenameToLoadModelFrom")); //$NON-NLS-1$
        }

        setModel(m_meta.getSerializedModelFileName());
      } else if (m_meta.getModel() != null) {
        // copy the primary model over to the data class
        // Copia o primeiro modelo sobre a classe de dados
        try {
          SerializedObject so = new SerializedObject(m_meta.getModel());
          WekaScoringModel defaultModel = (WekaScoringModel) so.getObject();

          m_data.setModel(defaultModel);
        } catch (Exception ex) {
          throw new KettleException(ex);
        }
      }

      // Check the input row meta data against the instances
      // Verifica a linha de entrada de meta dados contra a instancia.
      // header that the classifier was trained with
      // Cabecalho que a classificao  com quem foi treinado.
      try {
        Instances header = m_data.getModel().getHeader();
        m_data.mapIncomingRowMetaData(header, getInputRowMeta(),
            m_meta.getUpdateIncrementalModel(), log);
      } catch (Exception ex) {
        throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
            "WekaScoring.Error.IncomingDataFormatDoesNotMatchModel"), ex); //$NON-NLS-1$
      }

      // Determine the output format
      // Determina o formato da saida.
      m_meta.getFields(m_data.getOutputRowMeta(), getStepname(), null, null,
          this);

      if (!Const.isEmpty(m_meta.getBatchScoringSize())) {
        try {
          String bss = environmentSubstitute(m_meta.getBatchScoringSize());
          m_batchScoringSize = Integer.parseInt(bss);
        } catch (NumberFormatException ex) {
          String modelPreferred = environmentSubstitute(((BatchPredictor) m_meta
              .getModel().getModel()).getBatchSize());

          boolean sizeOk = false;
          if (!Const.isEmpty(modelPreferred)) {
            logBasic(BaseMessages.getString(WekaScoringMeta.PKG,
                "WekaScoring.Message.UnableToParseBatchScoringSize", //$NON-NLS-1$
                modelPreferred));
            try {
              m_batchScoringSize = Integer.parseInt(modelPreferred);
              sizeOk = true;
            } catch (NumberFormatException e) {
            }
          }

          if (!sizeOk) {
            logBasic(BaseMessages.getString(WekaScoringMeta.PKG,
                "WekaScoring.Message.UnableToParseBatchScoringSizeDefault", //$NON-NLS-1$
                WekaScoringMeta.DEFAULT_BATCH_SCORING_SIZE));

            m_batchScoringSize = WekaScoringMeta.DEFAULT_BATCH_SCORING_SIZE;
          }
        }
      }

      if (m_data.getModel().isBatchPredictor()) {
        m_batch = new ArrayList<Object[]>();
      }
    } // end (if first)

    // Make prediction for row using model
    // Faz uma previsao para a linha usando o modelo. 
    try {
      if (m_meta.getFileNameFromField()) {
        setModelFromField(r);
      }

      if (m_data.getModel().isBatchPredictor()
          && !m_meta.getFileNameFromField()) {
        try {
          // add current row to batch
          // Adicionar atual linha no lote
          m_batch.add(r);

          if (m_batch.size() == m_batchScoringSize) {
            outputBatchRows();
          }
        } catch (Exception ex) {
          throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
              "WekaScoring.Error.ErrorGettingBatchPredictions"), ex); //$NON-NLS-1$
        }
      } else {
        Object[] outputRow = m_data.generatePrediction(getInputRowMeta(),
            m_data.getOutputRowMeta(), r, m_meta);
        putRow(m_data.getOutputRowMeta(), outputRow);
      }
    } catch (Exception ex) {
      throw new KettleException(BaseMessages.getString(WekaScoringMeta.PKG,
          "WekaScoring.Error.UnableToMakePredictionForRow", getLinesRead()), ex); //$NON-NLS-1$
    }

    if (log.isRowLevel()) {
      log.logRowlevel(toString(), "Read row #" + getLinesRead() + " : " + r); //$NON-NLS-1$ //$NON-NLS-2$
    }

    if (checkFeedback(getLinesRead())) {
      logBasic("Linenr " + getLinesRead()); //$NON-NLS-1$
    }
    return true;
  }

  protected void outputBatchRows() throws Exception {
    // get predictions for the batch
    // Busca as previsaos para o lote
    Object[][] outputRows = m_data.generatePredictions(getInputRowMeta(),
        m_data.getOutputRowMeta(), m_batch, m_meta);

    if (log.isDetailed()) {
      logDetailed(BaseMessages.getString(WekaScoringMeta.PKG,
          "WekaScoring.Message.PredictingBatch")); //$NON-NLS-1$
    }

    // output the rows
    // Saida das linhas
    for (Object[] row : outputRows) {
      putRow(m_data.getOutputRowMeta(), row);
    }

    // reset batch
    // Limpa o lote
    m_batch.clear();
  }

  /**
   * Initialize the step.
   * 
   * PT-BR
   * 
   * Inicializa o step (passo)
   * 
   * @param smi a <code>StepMetaInterface</code> value
   * @param sdi a <code>StepDataInterface</code> value
   * @return a <code>boolean</code> value
   */
  @Override
  public boolean init(StepMetaInterface smi, StepDataInterface sdi) {
    m_meta = (WekaScoringMeta) smi;
    m_data = (WekaScoringData) sdi;

    if (super.init(smi, sdi)) {
      return true;
    }
    return false;
  }
}
