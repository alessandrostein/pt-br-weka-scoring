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

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;

import org.pentaho.di.core.CheckResult;
import org.pentaho.di.core.CheckResultInterface;
import org.pentaho.di.core.Const;
import org.pentaho.di.core.Counter;
import org.pentaho.di.core.annotations.Step;
import org.pentaho.di.core.database.DatabaseMeta;
import org.pentaho.di.core.exception.KettleException;
import org.pentaho.di.core.exception.KettleStepException;
import org.pentaho.di.core.exception.KettleXMLException;
import org.pentaho.di.core.row.RowMetaInterface;
import org.pentaho.di.core.row.ValueMeta;
import org.pentaho.di.core.row.ValueMetaInterface;
import org.pentaho.di.core.variables.VariableSpace;
import org.pentaho.di.core.variables.Variables;
import org.pentaho.di.core.xml.XMLHandler;
import org.pentaho.di.i18n.BaseMessages;
import org.pentaho.di.repository.ObjectId;
import org.pentaho.di.repository.Repository;
import org.pentaho.di.repository.kdr.KettleDatabaseRepository;
import org.pentaho.di.trans.Trans;
import org.pentaho.di.trans.TransMeta;
import org.pentaho.di.trans.step.BaseStepMeta;
import org.pentaho.di.trans.step.StepDataInterface;
import org.pentaho.di.trans.step.StepInterface;
import org.pentaho.di.trans.step.StepMeta;
import org.pentaho.di.trans.step.StepMetaInterface;
import org.w3c.dom.Node;

import weka.core.Instances;
import weka.core.SerializedObject;

/**
 * Contains the meta data for the WekaScoring step.
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}org)
 */
@Step(id = "WekaScoring", image = "WS.png", name = "Weka Scoring", description = "Appends predictions from a pre-built Weka model", categoryDescription = "Data Mining")
public class WekaScoringMeta extends BaseStepMeta implements StepMetaInterface {

    protected static Class<?> PKG = WekaScoringMeta.class;

    public static final String XML_TAG = "weka_scoring"; //$NON-NLS-1$

    /**
     * Use a model file specified in an incoming field Usa um arquivo modelo
     * especifico para um campo de entrada
     */
    private boolean m_fileNameFromField;

    /**
     * Whether to cache loaded models in memory (when they are being specified
     * by a field in the incoming rows
     *
     * Deve-se armazenar modelos carregados no chache de memoria (quando ele
     * estao a ser especificado por um campo nas linhas de entrada
     */
    private boolean m_cacheLoadedModels;

    /**
     * The name of the field that is being used to specify model file name/path
     * O nome do campo que esta sendo usado para especificar o nome/caminho do
     * arquivo modelo
     */
    private String m_fieldNameToLoadModelFrom;

    /**
     * File name of the serialized Weka model to load/import Nome do arquivo
     * serializado do modelo Weka para carregar/importar
     */
    private String m_modelFileName;

    /**
     * File name to save incrementally updated model to Nome do arquivo para
     * salvar o modelo incremental atualizado
     */
    private String m_savedModelFileName;

    /**
     * True if predicted probabilities are to be output (has no effect if the
     * class (target is numeric)
     *
     * Verdadeiro se as probabilidades previstas devem ser de saida(nao tem
     * efeito se e classe(alvo e numerico))
     */
    private boolean m_outputProbabilities;

    /**
     * True if user has selected to update a model on the incoming data stream
     * and the model supports incremental updates and there exists a column in
     * the incoming data stream that has been matched successfully to the class
     * attribute (if one exists).
     *
     * Verdadeiro se o usuario tenha selecionado para atualizar um modelo sobre
     * a entrada do fluxo de dados e o modelo incremental suportado atualizado e
     * existe uma coluna do fluxo de dados que tenha sido acompanhado com
     * sucesso para o atributo da classe (se um existir)
     */
    private boolean m_updateIncrementalModel;

    private boolean m_storeModelInStepMetaData;

    /**
     * Holds the actual Weka model (classifier or clusterer) Contem o modeo Weka
     * atual (classificador ou agrupador)
     */
    private WekaScoringModel m_model;

  // holds a default model - used only when model files are sourced
    // from a field in the incoming data rows. In this case, it is
    // the fallback model if there is no model file specified in the
    // current incoming row. Is also necessary so that getFields()
    // can determine the full output structure.
  // Contem um modelo padrao - usado apenas quando arquivos modelos sao originados
    // para um campo de linhas de entrada. Neste caso, o modelo retrocede se nao ha um 
    // arquivo modelo especificado na a linha de entrada atual. Este tambem necessario
    // entao que o modelo getFields() pode determinar para a estrutura completa de saida.
    private WekaScoringModel m_defaultModel;

    /**
     * Batch scoring size Tamanho do lote socoring
     */
    public static final int DEFAULT_BATCH_SCORING_SIZE = 100;
    private String m_batchScoringSize = ""; //$NON-NLS-1$

    public void setStoreModelInStepMetaData(boolean b) {
        m_storeModelInStepMetaData = b;
    }

    public boolean getStoreModelInStepMetaData() {
        return m_storeModelInStepMetaData;
    }

    /**
     * Set the batch size to use if the model is a batch scoring model
     *
     * PT-BR
     *
     * Define o tamanho do lote para usar se o modelo esta em um lote modelo
     * socoring.
     *
     * @param size the size of the batch to use o tamanho do lote para usar
     */
    public void setBatchScoringSize(String size) {
        m_batchScoringSize = size;
    }

    /**
     * Get the batch size to use if the model is a batch scoring model
     *
     * PT-BR
     *
     * Obtem o tamanho do lote para usar se o modelo esta em um lote modelo
     * scoring
     *
     * @return the size of the batch to use o tamanho do lote para usar
     */
    public String getBatchScoringSize() {
        return m_batchScoringSize;
    }

    /**
     * Creates a new <code>WekaScoringMeta</code> instance.
     *
     * Cria uma nova instancia de WekaScoringMeta
     */
    public WekaScoringMeta() {
        super(); // allocate BaseStepMeta
    }

    /**
     * Set whether filename is coming from an incoming field
     *
     * Define-se o nome preveniente de um campo de entrada.
     *
     * @param f true if the model to use is specified via path in an incoming
     * field value verdadeiro se o modelo usado e especificado via caminho do
     * campo de entrada
     */
    public void setFileNameFromField(boolean f) {
        m_fileNameFromField = f;
    }

    /**
     * Get whether filename is coming from an incoming field
     *
     * Resgata-se o nome proveniente de um campo de entrada
     *
     * @return true if the model to use is specified via path in an incoming
     * field value verdadeiro se o modelo usado e especificado via caminho do
     * campo de entrada
     */
    public boolean getFileNameFromField() {
        return m_fileNameFromField;
    }

    /**
     * Set whether to cache loaded models in memory
     *
     * Define-se o modelo cache carregado na memoria
     *
     * @param l true if models are to be cached in memory verdadeiro se o modelo
     * esta carregado no cache de memoria
     */
    public void setCacheLoadedModels(boolean l) {
        m_cacheLoadedModels = l;
    }

    /**
     * Get whether to cache loaded models in memory
     *
     * Resgata-se o modelo carregado no cache de memoria
     *
     * @return true if models are to be cached in memory verdadeiro se o modelo
     * esta carregado no cache de memoria
     */
    public boolean getCacheLoadedModels() {
        return m_cacheLoadedModels;
    }

    /**
     * Set the name of the incoming field that holds paths to model files
     *
     * Define o nome do campo de entrada que contem o caminho do arquivo modelo
     *
     * @param fn the name of the incoming field that holds model paths o nome do
     * campo de entrada que contem caminho do modelo
     */
    public void setFieldNameToLoadModelFrom(String fn) {
        m_fieldNameToLoadModelFrom = fn;
    }

    /**
     * Get the name of the incoming field that holds paths to model files
     *
     * Resgata o nome do campo de entrada que contem o caminho do arquivo modelo
     *
     * @return the name of the incoming field that holds model paths o nome do
     * campo de entrada que contem o caminho modelo
     */
    public String getFieldNameToLoadModelFrom() {
        return m_fieldNameToLoadModelFrom;
    }

    /**
     * Set the file name of the serialized Weka model to load/import from
     *
     * Define o nome do arquivo para o modelo Weka serializado para
     * carregar/importar
     *
     * @param mfile the file name o nome do arquivo
     */
    public void setSerializedModelFileName(String mfile) {
        m_modelFileName = mfile;
    }

    /**
     * Get the filename of the serialized Weka model to load/import from
     *
     * Retorna o nome do arquivo do modelo Weka serializado para
     * carregar/importar
     *
     * @return the file name o nome do arquivo
     */
    public String getSerializedModelFileName() {
        return m_modelFileName;
    }

    /**
     * Set the file name that the incrementally updated model will be saved to
     * when the current stream of data terminates
     *
     *
     * Define o nome do arquivo que atualiza de forma incremental o modelo para
     * ser salvado quando o fluxo de dados termina.
     *
     * @param savedM the file name to save to o nome do arquivo para salvar
     */
    public void setSavedModelFileName(String savedM) {
        m_savedModelFileName = savedM;
    }

    /**
     * Get the file name that the incrementally updated model will be saved to
     * when the current stream of data terminates
     *
     * Retorna o nome do arquivo que foi atualizado o modelo de forma
     * incremental para ser salvo quando o fluxo de dados termina.
     *
     * @return the file name to save to o nome do arquivo para salvar.
     */
    public String getSavedModelFileName() {
        return m_savedModelFileName;
    }

    /**
     * Set the Weka model
     *
     * Define o modelo Weka
     *
     * @param model a <code>WekaScoringModel</code> that encapsulates the actual
     * Weka model (Classifier or Clusterer) um WekaScoringModel que encapsula o
     * modelo Weka (Classificador ou agrupador) atual.
     */
    public void setModel(WekaScoringModel model) {
        m_model = model;
    }

    /**
     * Get the Weka model
     *
     * Retorna o modelo Weka
     *
     * @return a <code>WekaScoringModel</code> that encapsulates the actual Weka
     * model (Classifier or Clusterer) um WekaScoringModel que encapsula o
     * modelo Weka (Classificador ou agrupador) atual
     */
    public WekaScoringModel getModel() {
        return m_model;
    }

    /**
     * Gets the default model (only used when model file names are being sourced
     * from a field in the incoming rows).
     *
     * Retorna o modelo padrao (apenas usado quando o nome do arquivo modelo
     * estao sendo originados a partir de um campo nas linhas de entrada)
     *
     * @return the default model to use when there is no filename provided in
     * the incoming data row. o modelo padrao para usar quando nao ha um nome do
     * arquivo proveniente da linha de entrada.
     */
    public WekaScoringModel getDefaultModel() {
        return m_defaultModel;
    }

    /**
     * Sets the default model (only used when model file names are being sourced
     * from a field in the incoming rows).
     *
     * Define o modelo padrao (apenas usado quando o nome do arquivo modelo
     * estao sendo originados apartir de um campo nas linhas de entrada)
     *
     * @param defaultM the default model to use. o modelo padrao para usar
     */
    public void setDefaultModel(WekaScoringModel defaultM) {
        m_defaultModel = defaultM;
    }

    /**
     * Set whether to predict probabilities
     *
     * Define-se a probabilidade de previsao
     *
     * @param b true if a probability distribution is to be output verdadeiro se
     * a distribuicao de probabilidade e ser saida
     */
    public void setOutputProbabilities(boolean b) {
        m_outputProbabilities = b;
    }

    /**
     * Get whether to predict probabilities
     *
     * Resgata-o a probabilidade de previsao
     *
     * @return a true if a probability distribution is to be output verdadeiro
     * se a distribuicao de probabilidade e ser saida
     */
    public boolean getOutputProbabilities() {
        return m_outputProbabilities;
    }

    /**
     * Get whether the model is to be incrementally updated with each incoming
     * row (after making a prediction for it).
     *
     * Resgata-o o modelo que esta sendo atualizado de forma incremental com
     * cada linha de entrada (depois de fazer a previsao para ele)
     *
     * @return a true if the model is to be updated incrementally with each
     * incoming row verdadeorp se o modelo esta para ser atualizado de forma
     * incremental com cada linha de entrada
     */
    public boolean getUpdateIncrementalModel() {
        return m_updateIncrementalModel;
    }

    /**
     * Set whether to update the model incrementally
     *
     * Define-se para atualizar o modelo de forma incremental
     *
     * @param u true if the model should be updated with each incoming row
     * (after predicting it) verdadeiro se o modelo deve ser atualizado com cada
     * linha de entrada (depois da previsao dele)
     */
    public void setUpdateIncrementalModel(boolean u) {
        m_updateIncrementalModel = u;
    }

    protected String getXML(boolean logging) {
        StringBuffer retval = new StringBuffer(100);

        retval.append("<" + XML_TAG + ">"); //$NON-NLS-1$ //$NON-NLS-2$

        retval.append(XMLHandler.addTagValue("output_probabilities", //$NON-NLS-1$
                m_outputProbabilities));
        retval.append(XMLHandler.addTagValue("update_model", //$NON-NLS-1$
                m_updateIncrementalModel));
        retval.append(XMLHandler.addTagValue("store_model_in_meta", //$NON-NLS-1$
                m_storeModelInStepMetaData));

        if (m_updateIncrementalModel) {
      // any file name to save the changed model to?
            // Algum nome de arquivo para salvar o modelo alterado?
            if (!Const.isEmpty(m_savedModelFileName)) {
                retval.append(XMLHandler.addTagValue("model_export_file_name", //$NON-NLS-1$
                        m_savedModelFileName));
            }
        }

        retval.append(XMLHandler.addTagValue("file_name_from_field", //$NON-NLS-1$
                m_fileNameFromField));
        if (m_fileNameFromField) {
      // any non-null field name?
            // Algum nome do campo nao nulo?
            if (!Const.isEmpty(m_fieldNameToLoadModelFrom)) {
                retval.append(XMLHandler.addTagValue("field_name_to_load_from", //$NON-NLS-1$
                        m_fieldNameToLoadModelFrom));
                System.out.println(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.ModelSourcedFromField") //$NON-NLS-1$
                        + " " //$NON-NLS-1$
                        + m_fieldNameToLoadModelFrom);
            }
        }

        if (!Const.isEmpty(m_batchScoringSize)) {
            retval.append(XMLHandler.addTagValue("batch_scoring_size", //$NON-NLS-1$
                    m_batchScoringSize));
        }

        retval.append(XMLHandler.addTagValue("cache_loaded_models", //$NON-NLS-1$
                m_cacheLoadedModels));

        WekaScoringModel temp = (m_fileNameFromField) ? m_defaultModel : m_model;

    // can we save the model as XML?
        // Pode ser salvo o modelo em um XML?
        if (temp != null && Const.isEmpty(m_modelFileName)) {

            try {
        // Convert model to base64 encoding
                // Converte o modelo para codificacao base 64
                ByteArrayOutputStream bao = new ByteArrayOutputStream();
                BufferedOutputStream bos = new BufferedOutputStream(bao);
                ObjectOutputStream oo = new ObjectOutputStream(bos);
                oo.writeObject(temp);
                oo.flush();
                byte[] model = bao.toByteArray();
                String base64model = XMLHandler
                        .addTagValue("weka_scoring_model", model); //$NON-NLS-1$
                String modType = (m_fileNameFromField) ? "default" : ""; //$NON-NLS-1$ //$NON-NLS-2$
                System.out.println("Serializing " + modType + " model."); //$NON-NLS-1$ //$NON-NLS-2$
                System.out.println(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.SizeOfModel") + " " + base64model.length()); //$NON-NLS-1$ //$NON-NLS-2$

                retval.append(base64model);
                oo.close();
            } catch (Exception ex) {
                System.out.println(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.Base64SerializationProblem")); //$NON-NLS-1$
            }
        } else {
            if (!Const.isEmpty(m_modelFileName)) {

                if (logging) {
                    logDetailed(BaseMessages.getString(PKG,
                            "WekaScoringMeta.Log.ModelSourcedFromFile") + " " + m_modelFileName); //$NON-NLS-1$ //$NON-NLS-2$
                }
            }

      // save the model file name
            // Salva o nome do arquivo modelo
            retval.append(XMLHandler.addTagValue("model_file_name", m_modelFileName)); //$NON-NLS-1$
        }

        retval.append("</" + XML_TAG + ">"); //$NON-NLS-1$ //$NON-NLS-2$
        return retval.toString();
    }

    /**
     * Return the XML describing this (configured) step
     *
     * Retorna o XML descrevendo este (configuracao) step (passo)
     *
     * @return a <code>String</code> containing the XML uma String contendo o
     * XML
     */
    @Override
    public String getXML() {
        return getXML(true);
    }

    /**
     * Check for equality
     *
     * Verificar para igualdade
     *
     * @param obj an <code>Object</code> to compare with uma Objetivo para quem
     * comparar
     * @return true if equal to the supplied object retorna se igualdade com o
     * objeto visto
     */
    @Override
    public boolean equals(Object obj) {
        if (obj != null && (obj.getClass().equals(this.getClass()))) {
            WekaScoringMeta m = (WekaScoringMeta) obj;
            return (getXML(false) == m.getXML(false));
        }

        return false;
    }

    /**
     * Hash code method
     *
     * Metodo do codigo hash
     *
     * @return the hash code for this object o codigo de hash para este objeto
     */
    @Override
    public int hashCode() {
        return getXML(false).hashCode();
    }

    /**
     * Clone this step's meta data
     *
     * Copia este step's (passo) de dados de configuracaoes.
     *
     * @return the cloned meta data os dados de configuracaoes copiados
     */
    @Override
    public Object clone() {
        WekaScoringMeta retval = (WekaScoringMeta) super.clone();
    // deep copy the model (if any)
        // Copia 'profunda' o modeo (se algum)
        if (m_model != null) {
            try {
                SerializedObject so = new SerializedObject(m_model);
                WekaScoringModel copy = (WekaScoringModel) so.getObject();
                copy.setLog(getLog());
                retval.setModel(copy);
            } catch (Exception ex) {
                logError(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.DeepCopyingError")); //$NON-NLS-1$
            }
        }

    // deep copy the default model (if any)
        // Copia 'profunda' o modeo (se algum)
        if (m_defaultModel != null) {
            try {
                SerializedObject so = new SerializedObject(m_defaultModel);
                WekaScoringModel copy = (WekaScoringModel) so.getObject();
                copy.setLog(getLog());
                retval.setDefaultModel(copy);
            } catch (Exception ex) {
                logError(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.DeepCopyingError")); //$NON-NLS-1$
            }
        }

        return retval;
    }

    public void setDefault() {
        m_modelFileName = null;
        m_outputProbabilities = false;
    }

    /**
     * Loads the meta data for this (configured) step from XML.
     *
     * Configura as configuracoes para este (configurado) step (passo) para XML.
     *
     * @param stepnode the step to load o step(passo) carregado
     * @exception KettleXMLException if an error occurs se ocorrer um erro
     */
    public void loadXML(Node stepnode, List<DatabaseMeta> databases,
            Map<String, Counter> counters) throws KettleXMLException {
        int nrModels = XMLHandler.countNodes(stepnode, XML_TAG);

        if (nrModels > 0) {
            Node wekanode = XMLHandler.getSubNodeByNr(stepnode, XML_TAG, 0);

            String temp = XMLHandler.getTagValue(wekanode, "file_name_from_field"); //$NON-NLS-1$
            if (temp.equalsIgnoreCase("N")) { //$NON-NLS-1$
                m_fileNameFromField = false;
            } else {
                m_fileNameFromField = true;
            }

            if (m_fileNameFromField) {
                m_fieldNameToLoadModelFrom = XMLHandler.getTagValue(wekanode,
                        "field_name_to_load_from"); //$NON-NLS-1$
            }

            m_batchScoringSize = XMLHandler.getTagValue(wekanode,
                    "batch_scoring_size"); //$NON-NLS-1$

            String store = XMLHandler.getTagValue(wekanode, "store_model_in_meta"); //$NON-NLS-1$
            if (store != null) {
                m_storeModelInStepMetaData = store.equalsIgnoreCase("Y");
            }

            temp = XMLHandler.getTagValue(wekanode, "cache_loaded_models"); //$NON-NLS-1$
            if (temp.equalsIgnoreCase("N")) { //$NON-NLS-1$
                m_cacheLoadedModels = false;
            } else {
                m_cacheLoadedModels = true;
            }

      // try and get the XML-based model
            // Testar e pegar modelo XML-base
            boolean success = false;
            try {
                String base64modelXML = XMLHandler.getTagValue(wekanode,
                        "weka_scoring_model"); //$NON-NLS-1$

                deSerializeBase64Model(base64modelXML);
                success = true;

                String modType = (m_fileNameFromField) ? "default" : ""; //$NON-NLS-1$ //$NON-NLS-2$
                logBasic("Deserializing " + modType + " model."); //$NON-NLS-1$ //$NON-NLS-2$

                logDetailed(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.DeserializationSuccess")); //$NON-NLS-1$
            } catch (Exception ex) {
                success = false;
            }

            if (!success) {
        // fall back and try and grab a model file name
                // Retrocede e testa e chama um nome do arquivo modelo
                m_modelFileName = XMLHandler.getTagValue(wekanode, "model_file_name"); //$NON-NLS-1$
            }

            temp = XMLHandler.getTagValue(wekanode, "output_probabilities"); //$NON-NLS-1$
            if (temp.equalsIgnoreCase("N")) { //$NON-NLS-1$
                m_outputProbabilities = false;
            } else {
                m_outputProbabilities = true;
            }

            temp = XMLHandler.getTagValue(wekanode, "update_model"); //$NON-NLS-1$
            if (temp.equalsIgnoreCase("N")) { //$NON-NLS-1$
                m_updateIncrementalModel = false;
            } else {
                m_updateIncrementalModel = true;
            }

            if (m_updateIncrementalModel) {
                m_savedModelFileName = XMLHandler.getTagValue(wekanode,
                        "model_export_file_name"); //$NON-NLS-1$
            }
        }

    // check the model status. If no model and we have
        // a file name, try and load here. Otherwise, loading
        // wont occur until the transformation starts or the
        // user opens the configuration gui in Spoon. This affects
        // the result of the getFields method and has an impact
        // on downstream steps that need to know what we produce
    // Verifica o status do modelo. Se nenhum modelo e temos
        // um nome de arquivo e tentar carregar aqui. Senao, o carregamento
        // nao vai ocorrer ate que a transformacao comecou ou o usuario configura
        // diretamente na interface do Spoon. Isso afeta os resultados para o metodo
        // getFields e tem um impacto sobre as medidas que necessitam saber o que
        // nos produzimos.
        /*
         * WekaScoringModel temp = (m_fileNameFromField) ? m_defaultModel : m_model;
         * if (temp == null && !Const.isEmpty(m_modelFileName)) { try {
         * loadModelFile(); } catch (Exception ex) { throw new
         * KettleXMLException(BaseMessages.getString(PKG,
         * "WekaScoring.Error.ProblemDeserializingModel"), ex); //$NON-NLS-1$ } }
         */
    }

    protected void loadModelFile() throws Exception {
        /*
         * File modelFile = new File(m_modelFileName); if (modelFile.exists()) {
         */
        if (WekaScoringData.modelFileExists(m_modelFileName, new Variables())) {
            if (m_fileNameFromField) {
                logDetailed(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Message.LoadingDefaultModelFromFile")); //$NON-NLS-1$
                m_defaultModel = WekaScoringData.loadSerializedModel(m_modelFileName,
                        getLog(), new Variables());
            } else {
                logDetailed(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Message.LoadingModelFromFile")); //$NON-NLS-1$
                m_model = WekaScoringData.loadSerializedModel(m_modelFileName,
                        getLog(), new Variables());
            }
        }
    }

    protected void deSerializeBase64Model(String base64modelXML) throws Exception {
        byte[] model = XMLHandler.stringToBinary(base64modelXML);

        // now de-serialize
        ByteArrayInputStream bis = new ByteArrayInputStream(model);
        ObjectInputStream ois = new ObjectInputStream(bis);

        if (m_fileNameFromField) {
            m_defaultModel = (WekaScoringModel) ois.readObject();
        } else {
            m_model = (WekaScoringModel) ois.readObject();
        }
        ois.close();
    }

    /**
     * Read this step's configuration from a repository
     *
     * Leia esta configuracao step (passo) de um repositorio
     *
     * @param rep the repository to access o repositorio para acessar
     * @param id_step the id for this step o identificador deste step (passo)
     * @exception KettleException if an error occurs se ocorrer um erro
     */
    public void readRep(Repository rep, ObjectId id_step,
            List<DatabaseMeta> databases, Map<String, Counter> counters)
            throws KettleException {
        m_fileNameFromField = rep.getStepAttributeBoolean(id_step, 0,
                "file_name_from_field"); //$NON-NLS-1$

        m_batchScoringSize = rep.getStepAttributeString(id_step, 0,
                "batch_scoring_size"); //$NON-NLS-1$

        if (m_fileNameFromField) {
            m_fieldNameToLoadModelFrom = rep.getStepAttributeString(id_step, 0,
                    "field_name_to_load_from"); //$NON-NLS-1$
        }

        m_cacheLoadedModels = rep.getStepAttributeBoolean(id_step, 0,
                "cache_loaded_models"); //$NON-NLS-1$

        m_storeModelInStepMetaData = rep.getStepAttributeBoolean(id_step, 0,
                "store_model_in_meta"); //$NON-NLS-1$

    // try and get a filename first as this overrides any model stored
        // in the repository
    // Testar e buscar o nome do arquivo primeiro para substituir qualquer modelo
        // armaznado no repositorio.
        boolean success = false;
        try {
            m_modelFileName = rep.getStepAttributeString(id_step, 0,
                    "model_file_name"); //$NON-NLS-1$
            success = true;
            if (m_modelFileName == null || Const.isEmpty(m_modelFileName)) {
                success = false;
            }
        } catch (KettleException ex) {
            success = false;
        }

        if (!success) {
      // try and get the model itself...
            // Testa e buscar o modelo se ...
            try {
                String base64XMLModel = rep.getStepAttributeString(id_step, 0,
                        "weka_scoring_model"); //$NON-NLS-1$
                logDebug(BaseMessages.getString(PKG, "WekaScoringMeta.Log.SizeOfModel") //$NON-NLS-1$
                        + " " + base64XMLModel.length()); //$NON-NLS-1$

                if (base64XMLModel != null && base64XMLModel.length() > 0) {
                    // try to de-serialize
                    deSerializeBase64Model(base64XMLModel);
                    success = true;
                } else {
                    success = false;
                }
            } catch (Exception ex) {
                ex.printStackTrace();
                success = false;
            }
        }

        m_outputProbabilities = rep.getStepAttributeBoolean(id_step, 0,
                "output_probabilities"); //$NON-NLS-1$

        m_updateIncrementalModel = rep.getStepAttributeBoolean(id_step, 0,
                "update_model"); //$NON-NLS-1$

        if (m_updateIncrementalModel) {
            m_savedModelFileName = rep.getStepAttributeString(id_step, 0,
                    "model_export_file_name"); //$NON-NLS-1$
        }

    // check the model status. If no model and we have
        // a file name, try and load here. Otherwise, loading
        // wont occur until the transformation starts or the
        // user opens the configuration gui in Spoon. This affects
        // the result of the getFields method and has an impact
        // on downstream steps that need to know what we produce
    // Verifica o status do modelo. Se nenhum modelo e temos
        // um nome de arquivo e tentar carregar aqui. Senao, o carregamento
        // nao vai ocorrer ate que a transformacao comecou ou o usuario configura
        // diretamente na interface do Spoon. Isso afeta os resultados para o metodo
        // getFields e tem um impacto sobre as medidas que necessitam saber o que
        // nos produzimos.
        /*
         * WekaScoringModel temp = (m_fileNameFromField) ? m_defaultModel : m_model;
         * if (temp == null && !Const.isEmpty(m_modelFileName)) { try {
         * loadModelFile(); } catch (Exception ex) { throw new
         * KettleException(BaseMessages.getString(PKG,
         * "WekaScoring.Error.ProblemDeserializingModel"), ex); //$NON-NLS-1$ } }
         */
    }

    /**
     * Save this step's meta data to a repository
     *
     * Salva este step's (passo) de configuracoes em um repositorio
     *
     * @param rep the repository to save to o repositorio para salvar
     * @param id_transformation transformation id indentificador da
     * transformacao
     * @param id_step step id o identificador do step (passo)
     * @exception KettleException if an error occurs se ocorrer um erro
     */
    public void saveRep(Repository rep, ObjectId id_transformation,
            ObjectId id_step) throws KettleException {

        rep.saveStepAttribute(id_transformation, id_step, 0,
                "output_probabilities", m_outputProbabilities); //$NON-NLS-1$

        rep.saveStepAttribute(id_transformation, id_step, 0, "update_model", //$NON-NLS-1$
                m_updateIncrementalModel);

        if (m_updateIncrementalModel) {
            // any file name to save the changed model to?
            // Algum nome de arquivo para salvar o modelo alterado?
            if (!Const.isEmpty(m_savedModelFileName)) {
                rep.saveStepAttribute(id_transformation, id_step, 0,
                        "model_export_file_name", m_savedModelFileName); //$NON-NLS-1$
            }
        }

        rep.saveStepAttribute(id_transformation, id_step, 0,
                "file_name_from_field", m_fileNameFromField); //$NON-NLS-1$
        if (m_fileNameFromField) {
            rep.saveStepAttribute(id_transformation, id_step, 0,
                    "field_name_to_load_from", m_fieldNameToLoadModelFrom); //$NON-NLS-1$
        }

        rep.saveStepAttribute(id_transformation, id_step, 0, "cache_loaded_models", //$NON-NLS-1$
                m_cacheLoadedModels);

        rep.saveStepAttribute(id_transformation, id_step, 0, "store_model_in_meta", //$NON-NLS-1$
                m_storeModelInStepMetaData);

        if (!Const.isEmpty(m_batchScoringSize)) {
            rep.saveStepAttribute(id_transformation, id_step, 0,
                    "batch_scoring_size", m_batchScoringSize); //$NON-NLS-1$
        }

        WekaScoringModel temp = (m_fileNameFromField) ? m_defaultModel : m_model;

        if (temp != null && Const.isEmpty(m_modelFileName)) {
            try {
                // Convert model to base64 encoding
                // Converte o modelo para codificacao base64
                ByteArrayOutputStream bao = new ByteArrayOutputStream();
                BufferedOutputStream bos = new BufferedOutputStream(bao);
                ObjectOutputStream oo = new ObjectOutputStream(bos);
                oo.writeObject(temp);
                oo.flush();
                byte[] model = bao.toByteArray();
                String base64XMLModel = KettleDatabaseRepository
                        .byteArrayToString(model);

                String modType = (m_fileNameFromField) ? "default" : ""; //$NON-NLS-1$ //$NON-NLS-2$
                logDebug("Serializing " + modType + " model."); //$NON-NLS-1$ //$NON-NLS-2$

                rep.saveStepAttribute(id_transformation, id_step, 0,
                        "weka_scoring_model", base64XMLModel); //$NON-NLS-1$
                oo.close();
            } catch (Exception ex) {
                logError(BaseMessages.getString(PKG,
                        "WekaScoringDialog.Log.Base64SerializationProblem"), ex); //$NON-NLS-1$
            }
        } else {
            // either XStream is not present or user wants to source from
            // file
            
            // XStream nao esta presente ou o usuario quer a fonte do arquivo
            if (!Const.isEmpty(m_modelFileName)) {
                logBasic(BaseMessages.getString(PKG,
                        "WekaScoringMeta.Log.ModelSourcedFromFile") + " " + m_modelFileName); //$NON-NLS-1$ //$NON-NLS-2$
            }

            rep.saveStepAttribute(id_transformation, id_step, 0, "model_file_name", //$NON-NLS-1$
                    m_modelFileName);
        }
    }

    /**
     * Generates row meta data to represent the fields output by this step
     * 
     * Gera dados da linha de configuracoes para representar os campos de 
     * saida pelo step (passo)
     *
     * @param row the meta data for the output produced
     *            o dado de configuracao para produzir a saida
     * @param origin the name of the step to be used as the origin
     *               o nome do step (passo) para ser usado como original
     * @param info The input rows metadata that enters the step through the
     * specified channels in the same order as in method getInfoSteps(). The
     * step metadata can then choose what to do with it: ignore it or not.
     *             As configuracoes de entrada que entra no step (passo)  
     * atraves dos canais especificados na mesmo ordem que no metodo getInfoSteps().As
     * configuracoes do step (passo) podem entao escolher o que fazer com ele: ignorar ou nao.
     * @param nextStep if this is a non-null value, it's the next step in the
     * transformation. The one who's asking, the step where the data is
     * targetted towards.
     *                 se este nao e um valor nulo, e o proximo step (passo) na transformacao. A
     * aplicacao esta pedindo, a etapa onde os dados serao direcionados.
     * @param space not sure what this is :-)
     *              nao tenho certeza o que isto e kkkk
     * @exception KettleStepException if an error occurs
     *                                se ocorrer um erro
     */
    @Override
    public void getFields(RowMetaInterface row, String origin,
            RowMetaInterface[] info, StepMeta nextStep, VariableSpace space)
            throws KettleStepException {

        if (m_model == null && !Const.isEmpty(getSerializedModelFileName())) {
            // see if we can load from a file.

            String modName = getSerializedModelFileName();

            // if (!modelFile.exists()) {
            try {
                if (!WekaScoringData.modelFileExists(modName, space)) {
                    throw new KettleStepException(BaseMessages.getString(PKG,
                            "WekaScoring.Error.NonExistentModelFile")); //$NON-NLS-1$
                }

                WekaScoringModel model = WekaScoringData.loadSerializedModel(
                        m_modelFileName, getLog(), space);
                setModel(model);
            } catch (Exception ex) {
                throw new KettleStepException(BaseMessages.getString(PKG,
                        "WekaScoring.Error.ProblemDeserializingModel"), ex); //$NON-NLS-1$
            }
        }

        if (m_model != null) {
            Instances header = m_model.getHeader();
            String classAttName = null;
            boolean supervised = m_model.isSupervisedLearningModel();

            if (supervised) {
                classAttName = header.classAttribute().name();

                if (header.classAttribute().isNumeric() || !m_outputProbabilities) {
                    int valueType = (header.classAttribute().isNumeric()) ? ValueMetaInterface.TYPE_NUMBER
                            : ValueMetaInterface.TYPE_STRING;

                    ValueMetaInterface newVM = new ValueMeta(classAttName + "_predicted", //$NON-NLS-1$
                            valueType);
                    newVM.setOrigin(origin);
                    row.addValueMeta(newVM);
                } else {
                    for (int i = 0; i < header.classAttribute().numValues(); i++) {
                        String classVal = header.classAttribute().value(i);
                        ValueMetaInterface newVM = new ValueMeta(classAttName + ":" //$NON-NLS-1$
                                + classVal + "_predicted_prob", ValueMetaInterface.TYPE_NUMBER); //$NON-NLS-1$
                        newVM.setOrigin(origin);
                        row.addValueMeta(newVM);
                    }
                }
            } else {
                if (m_outputProbabilities) {
                    try {
                        int numClusters = ((WekaScoringClusterer) m_model)
                                .numberOfClusters();
                        for (int i = 0; i < numClusters; i++) {
                            ValueMetaInterface newVM = new ValueMeta("cluster_" + i //$NON-NLS-1$
                                    + "_predicted_prob", ValueMetaInterface.TYPE_NUMBER); //$NON-NLS-1$
                            newVM.setOrigin(origin);
                            row.addValueMeta(newVM);
                        }
                    } catch (Exception ex) {
                        throw new KettleStepException(BaseMessages.getString(PKG,
                                "WekaScoringMeta.Error.UnableToGetNumberOfClusters"), ex); //$NON-NLS-1$
                    }
                } else {
                    ValueMetaInterface newVM = new ValueMeta("cluster#_predicted", //$NON-NLS-1$
                            ValueMetaInterface.TYPE_NUMBER);
                    newVM.setOrigin(origin);
                    row.addValueMeta(newVM);
                }
            }
        }
    }

    /**
     * Check the settings of this step and put findings in a remarks list.
     *
     * @param remarks the list to put the remarks in. see
     * <code>org.pentaho.di.core.CheckResult</code>
     * @param transmeta the transform meta data
     * @param stepMeta the step meta data
     * @param prev the fields coming from a previous step
     * @param input the input step names
     * @param output the output step names
     * @param info the fields that are used as information by the step
     */
    public void check(List<CheckResultInterface> remarks, TransMeta transmeta,
            StepMeta stepMeta, RowMetaInterface prev, String[] input,
            String[] output, RowMetaInterface info) {

        CheckResult cr;

        if ((prev == null) || (prev.size() == 0)) {
            cr = new CheckResult(CheckResult.TYPE_RESULT_WARNING,
                    "Not receiving any fields from previous steps!", stepMeta); //$NON-NLS-1$
            remarks.add(cr);
        } else {
            cr = new CheckResult(CheckResult.TYPE_RESULT_OK,
                    "Step is connected to previous one, receiving " + prev.size() //$NON-NLS-1$
                    + " fields", stepMeta); //$NON-NLS-1$
            remarks.add(cr);
        }

        // See if we have input streams leading to this step!
        if (input.length > 0) {
            cr = new CheckResult(CheckResult.TYPE_RESULT_OK,
                    "Step is receiving info from other steps.", stepMeta); //$NON-NLS-1$
            remarks.add(cr);
        } else {
            cr = new CheckResult(CheckResult.TYPE_RESULT_ERROR,
                    "No input received from other steps!", stepMeta); //$NON-NLS-1$
            remarks.add(cr);
        }

        if (m_model == null) {
            if (!Const.isEmpty(m_modelFileName)) {
                File f = new File(m_modelFileName);
                if (!f.exists()) {
                    cr = new CheckResult(CheckResult.TYPE_RESULT_ERROR,
                            "Step does not have access to a " + "usable model!", stepMeta); //$NON-NLS-1$ //$NON-NLS-2$
                    remarks.add(cr);
                }
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.pentaho.di.trans.step.BaseStepMeta#getDialogClassName()
     */
    @Override
    public String getDialogClassName() {
        return "org.pentaho.di.scoring.WekaScoringDialog"; //$NON-NLS-1$
    }

    /**
     * Get the executing step, needed by Trans to launch a step.
     *
     * @param stepMeta the step info
     * @param stepDataInterface the step data interface linked to this step.
     * Here the step can store temporary data, database connections, etc.
     * @param cnr the copy number to get.
     * @param tr the transformation info.
     * @param trans the launching transformation
     * @return a <code>StepInterface</code> value
     */
    public StepInterface getStep(StepMeta stepMeta,
            StepDataInterface stepDataInterface, int cnr, TransMeta tr, Trans trans) {

        return new WekaScoring(stepMeta, stepDataInterface, cnr, tr, trans);
    }

    /**
     * Get a new instance of the appropriate data class. This data class
     * implements the StepDataInterface. It basically contains the persisting
     * data that needs to live on, even if a worker thread is terminated.
     *
     * @return a <code>StepDataInterface</code> value
     */
    public StepDataInterface getStepData() {

        return new WekaScoringData();
    }
}
