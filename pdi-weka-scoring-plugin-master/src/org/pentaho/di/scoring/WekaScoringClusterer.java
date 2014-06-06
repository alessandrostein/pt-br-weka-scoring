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

import weka.core.BatchPredictor;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.clusterers.Clusterer;
import weka.clusterers.UpdateableClusterer;
import weka.clusterers.DensityBasedClusterer;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Subclass of WekaScoringModel that encapsulates a Clusterer.
 * 
 * PT-BR
 * 
 * Classe filha de WekaScoringModel que encapsula um Clusterer
 *
 * @author  Mark Hall (mhall{[at]}pentaho.org)
 * @version 1.0
 */
class WekaScoringClusterer extends WekaScoringModel {
  
  // The encapsulated clusterer
  // O Clusterer encapsulado
  private Clusterer m_model;

  // Any attributes to ignore
  // Alguns atributos sao ignorados
  private Remove m_ignoredAtts;

  private String m_ignoredString;
  
  /**
   * Creates a new <code>WekaScoringClusterer</code> instance.
   * 
   * PT-BR
   * 
   * Cria uma nova instancia de WekaScoringClusterer
   *
   * @param model the Clusterer
   */
  public WekaScoringClusterer(Object model) {
    super(model);
  }

  /**
   * Sets up a Remove filter to remove attributes that
   * are to be ignored by the clusterer. setHeader must
   * be called before this method.
   * 
   * PT-BR
   * 
   * Configura uma remocao de filtro para remover atributos que
   * sao ignorados pelo clusterer. O metodo setHeader necessario
   * ser chamado antes deste metodo.
   *
   * @param attsToIgnore any attributes to ignore during the scoring process
   *                     Qualquer atributo ignorado durante o processo scoring
   */
  public void setAttributesToIgnore(int[] attsToIgnore) throws Exception {
    Instances headerI = getHeader();
    m_ignoredAtts = new Remove();
    m_ignoredAtts.setAttributeIndicesArray(attsToIgnore);
    m_ignoredAtts.setInvertSelection(false);
    m_ignoredAtts.setInputFormat(headerI);

    StringBuffer temp = new StringBuffer();
    temp.append("Attributes ignored by clusterer:\n\n");
    for (int i = 0; i < attsToIgnore.length; i++) {
      temp.append(headerI.attribute(attsToIgnore[i]).name() + "\n");
    }
    temp.append("\n\n");
    m_ignoredString = temp.toString();
  }
  
  /**
   * Set the Clusterer model
   * 
   * PT-BR
   * 
   * Define o modelo do Clusterer
   *
   * @param model a Clusterer
   */
  public void setModel(Object model) {
      m_model = (Clusterer)model;
  }

  /**
   * Get the weka model
   * 
   * PT-BR
   * 
   * Retorna o modelo Weka
   *
   * @return the Weka model as an object
   *         O modelo Weka como objeto  
   */
  public Object getModel() {
    return m_model;
  }
  
  /**
   * Return a classification (cluster that the test instance
   * belongs to)
   * 
   * PT-BR
   * 
   * Retorna uma classificao (cluster que a instancia de teste 
   * pertece)
   *
   * @param inst the Instance to be clustered (predicted)
   *             a Instancia a ser agrupada (prevista)
   * @return the cluster number
   *         o numero agrupado
   * @exception Exception if an error occurs
   *            Exceção se ocorrer um erro
   */
  public double classifyInstance(Instance inst) throws Exception {
    if (m_ignoredAtts != null) {
      inst = applyFilter(inst);
    }
    return (double)m_model.clusterInstance(inst);
  }

  /**
   * Update (if possible) the model with the supplied instance
   * 
   * PT-BR
   * 
   * Atualiza (se possivel) o modelo fornecido pela instancia.
   *
   * @param inst the Instance to update with
   *             a Istance para atualizar
   * @return true if the update was updated successfully
   *              se a atualizacao ocorreu com sucesso
   * @exception Exception if an error occurs
   *            Exceção se ocorrer um erro
   */
  public boolean update(Instance inst) throws Exception {
    // Only cobweb is updateable at present
    // So cobweb(?) e atualizado no momento
    if (isUpdateableModel()) {
      if (m_ignoredAtts != null) {
        inst = applyFilter(inst);
      }
      //      System.err.println("In update...");
      ((UpdateableClusterer)m_model).updateClusterer(inst);
      //      System.err.println(m_model);
      return true;
    }
    return false;
  }
  
  /**
   * Return a probability distribution (over clusters).
   * 
   * PT-BR
   * 
   * Retorna a probabilidade de distribuicao (sobre agrupamento)
   *
   * @param inst the Instance to be predicted
   *             a Instancia a ser prevista 
   * @return a probability distribution
   *         a probabilidade de distribuicao
   * @exception Exception if an error occurs
   *            Exceção se ocorrer um erro
   */  
  public double[] distributionForInstance(Instance inst)
    throws Exception {
    if (m_ignoredAtts != null) {
      inst = applyFilter(inst);
    }
    return m_model.distributionForInstance(inst);
  }

  private Instance applyFilter(Instance inputInstance) throws Exception {
    if (!m_ignoredAtts.input(inputInstance)) {
      throw new Exception("[WekaScoring] Filter didn't make the test instance"
                          + " immediately available!");
    }
    m_ignoredAtts.batchFinished();
    Instance newInstance = m_ignoredAtts.output();
    return newInstance;
  }

  /**
   * Returns false. Clusterers are unsupervised methods.
   * 
   * PT-BR
   * 
   * Retorna falso. Clusterers sao metodos sem supervisao
   *
   * @return false
   */
  
  public boolean isSupervisedLearningModel() {
    return false;
  }

  /**
   * Returns false. No clusterers in Weka are 
   * incremental... yet
   * 
   * PT-BR
   * 
   * Retorna falso. Nao agrupado no Weka sao 
   * incremental... ainda
   *
   * @return false
   */
  public boolean isUpdateableModel() {
    if (m_model instanceof UpdateableClusterer) {
      return true;
    }
    
    return false;
  }

  /**
   * Returns true if the wrapped clusterer can produce
   * cluster membership probability estimates
   * 
   * PT-BR
   * 
   * Retorna verdadeiro se o agrupamento embrulhado pode produzir
   * estimativas de probabilidade de associacao ao agrupamento.
   * 
   * @return true if probability estimates can be produced
   *              se estimativa de probabilidade pode ser produzido
   */
  public boolean canProduceProbabilities() {
    if (m_model instanceof DensityBasedClusterer) {
      return true;
    }
    return false;
  }

  /**
   * Returns the number of clusters that the encapsulated
   * Clusterer has learned.
   * 
   * Retorna o numero do agrupamento que e encapsulado
   * Clusterer aprendeu
   *
   * @return the number of clusters in the model.
   *         o numero de agrupamento no modelo   
   * @exception Exception if an error occurs
   *            Excecao se ocorrer um erro
   */
  public int numberOfClusters() throws Exception {
    return m_model.numberOfClusters();
  }

  /**
   * Returns the textual description of the Clusterer's model.
   *
   * PT-BR
   * 
   * Retorna uma descricao textual do modelo do Clusterer
   * 
   * @return the Clusterer's model as a String
   *         o modelo Clusterer como uma String 
   */
  public String toString() {
    String ignored = (m_ignoredString == null)
      ? ""
      : m_ignoredString;

    return ignored + m_model.toString();
  }

  /**
   * Batch scoring method.
   * 
   * PT-BR
   * 
   * Lote do metodo scoring
   * 
   * @param insts the instances to score
   *              a instancia para marcar
   * @return an array of predictions (index of the predicted class label for
   * each instance)
   *         uma matriz de previsoes (indice do label de classe previsto para
   * cada instancia)
   * @throws Exception if a problem occurs
   *                   se ocorrer um erro
   */
  public double[] classifyInstances(Instances insts) throws Exception {
    double[][] preds = distributionsForInstances(insts);
    
    double[] result = new double[preds.length];
    for (int i = 0; i < preds.length; i++) {
      double[] p = preds[i];
      
      if (Utils.sum(p) <= 0) {
        result[i] = Utils.missingValue();
      } else {      
        result[i] = Utils.maxIndex(p);
      }
    }
    
    return result;
  }

  /**
   * Batch scoring method
   * 
   * PT-BR
   * 
   * Fornada de metodos scoring
   * 
   * @param insts the instances to get predictions for
   *              a instancia para obter previsoes 
   * @return an array of probability distributions, one for each instance
   *         uma matriz de prababilidade de distribuicao, uma para cada instancia
   * @throws Exception if a problem occurs
   *                   se ocorrer um erro
   */
  public double[][] distributionsForInstances(Instances insts) throws Exception {
    if (!isBatchPredictor()) {
      throw new Exception("Weka model cannot produce batch predictions!");
    }
    
    return ((BatchPredictor)m_model).distributionsForInstances(insts);
  }

  /**
   * Returns true if the encapsulated Weka model can produce 
   * predictions in a batch.
   * 
   * PT-BR
   * 
   * Retorna verdadeiro se o encapsulamento do modelo Weka pode produzir
   * previsoes em um lote.
   * 
   * @return true if the encapsulated Weka model can produce 
   * predictions in a batch
   *              se o encapsulamento do modelo Weka pode produzir
   * previsoes em um lote.
   */
  public boolean isBatchPredictor() {
    return (m_model instanceof BatchPredictor);
  }
}
