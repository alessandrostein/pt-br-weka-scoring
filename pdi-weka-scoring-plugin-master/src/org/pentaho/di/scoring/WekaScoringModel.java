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

import java.io.Serializable;

import org.pentaho.di.core.logging.LogChannelInterface;
import org.pentaho.dm.commons.LogAdapter;

import weka.classifiers.Classifier;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.PMMLModel;

/**
 * Abstract wrapper class for a Weka model. Provides a unified interface to
 * obtaining predictions. Subclasses ( WekaScoringClassifer and
 * WekaScoringClusterer) encapsulate the actual weka models.
 * 
 * Classe abstrata empacotada para o modelo Weka. Fornece uma interface unificada para
 * obter previsoes. Subclasses (WekaScoringClassifer e WekaScoringClusterer) encapsula
 * o modelo weka atual.
 * 
 * @author Mark Hall (mhall{[at]}pentaho.org)
 * @version 1.0
 */
public abstract class WekaScoringModel implements Serializable {

  // The header of the Instances used to build the model
  // O cabecalho de Instaces usado para construir o modelo
  private Instances m_header;

  /**
   * Creates a new <code>WekaScoringModel</code> instance.
   * 
   * PT-BR
   * 
   * Cria uma nova instancia de WekaScoringModel
   * 
   * @param model the actual Weka model to enacpsulate
   *              o modelo weka atual encapsulado. 
   */
  public WekaScoringModel(Object model) {
    setModel(model);
  }

  /**
   * Set the log to pass on to the model. Only PMML models require logging.
   * 
   * PT-BR
   * 
   * Define o log para passar para o modelo. Apenas o modelo PMML requer log.
   * 
   * @param log the log to use
   *            o log em uso. 
   */
  public void setLog(LogChannelInterface log) {
    if (getModel() instanceof PMMLModel) {
      LogAdapter logger = new LogAdapter(log);
      ((PMMLModel) getModel()).setLog(logger);
    }
  }

  /**
   * Set the Instances header
   * 
   * PT-BR
   * 
   * Define o cabecalho da Instances
   * 
   * @param header an <code>Instances</code> value
   *               um valor de Instance
   */
  public void setHeader(Instances header) {
    m_header = header;
  }

  /**
   * Get the header of the Instances that was used build this Weka model
   * 
   * PT-BR
   * 
   * Busca o cabecalho da Intances que foi usado para construir o modelo Weka
   * 
   * @return an <code>Instances</code> value
   *         um valor de Instances
   */
  public Instances getHeader() {
    return m_header;
  }

  /**
   * Tell the model that this scoring run is finished.
   * 
   * PT-BR
   * 
   * Dizer que este modelo scoring foi finalizado a execucao.
   */
  public void done() {
    // subclasses override if they need to do
    // something here.
      
    // Subclasses sobreescreve se eles precisar fazer alguma coisa aqui.
  }

  /**
   * Set the weka model
   * 
   * PT-BR
   * 
   * Define o modelo Weka
   * 
   * @param model the Weka model
   *              o modelo Weka
   */
  public abstract void setModel(Object model);

  /**
   * Get the weka model
   * 
   * PT-BR
   * 
   * Busca o modelo Weka
   * 
   * @return the Weka model as an object
   *         o modelo Weka como um Objeto
   */
  public abstract Object getModel();

  /**
   * Return a classification. What this represents depends on the implementing
   * sub-class. It could be the index of a class-value, a numeric value or a
   * cluster number for example.
   * 
   * PT-BR
   * 
   * Retorna a classificao. O que isto representa depende da implementacao
   * da sub-classe. Poderia ser o indice do valor da classe, um valor numerico ou 
   * um grupo de numeroos por exemplo
   * 
   * @param inst the Instance to be classified (predicted)
   *             a Instance para ser classificada (prevista)
   * @return the prediction
   *         a previsao 
   * @exception Exception if an error occurs
   *                      se ocorrer um erro
   */
  public abstract double classifyInstance(Instance inst) throws Exception;

  /**
   * Return a probability distribution (over classes or clusters).
   * 
   * PT-BR
   * 
   * Retorna uma distribucao de probabilidade (atraves de classes ou grupos)
   * 
   * @param inst the Instance to be predicted
   *             a Instance para ser prevista
   * @return a probability distribution 
   *         uma distribuicao de probabilidade
   * @exception Exception if an error occurs
   *                      se ocorrer um erro
   */
  public abstract double[] distributionForInstance(Instance inst)
      throws Exception;

  /**
   * Batch scoring method. Call isBatchPredictor() first in order to determine
   * if the underlying model can handle batch scoring.
   * 
   * PT-BR 
   * 
   * Metodo de lote scoring. Chama o metodo isBatchPredictor() primeiramente em 
   * ordem para determinar se o modelo subjacente pode manusear o lote scoring.
   * 
   * @param insts the instances to score
   *              a instancia para score.
   * @return an array of predictions
   *         uma matriz de previsoes
   * @throws Exception if a problem occurs
   *                   se ocorre um problema
   */
  public abstract double[] classifyInstances(Instances insts) throws Exception;

  /**
   * Batch scoring method. Call isBatchPredictor() first in order to determine
   * if the underlying model can handle batch scoring.
   * 
   * PT-BR
   * 
   * Metodo de lote scoring. Chama o metodo isBatchPredictor() primeiramente em
   * ordem para determinar se o modelo subjacente pode manusear o lote scoring.
   * 
   * @param insts the instances to score
   *              a instancia para score
   * @return an array of probability distributions, one for each instance
   *         uma matriz de distribuicao de probabilidade, uma para cada instancia.
   * @throws Exception if a problem occurs
   *                   se ocorrer um problema
   */
  public abstract double[][] distributionsForInstances(Instances insts)
      throws Exception;

  /**
   * Returns true if the encapsulated Weka model is a supervised model (i.e. has
   * been built to predict a single target in the data).
   * 
   * PT-BR
   * 
   * Retorna verdadeiro se o modelo Wka encapsulado e um modelo supervisionado.
   * 
   * @return true if the encapsulated Weka model is a supervised model
   *              se o modelo Weka encapsulado e um modelo supervisionado
   */
  public abstract boolean isSupervisedLearningModel();

  /**
   * Returns true if the encapsulated Weka model can be updated incrementally in
   * an instance by instance fashion.
   * 
   * PT-BR
   * 
   * Retorna verdadeiro se o modelo Weka encapsulado pode atualizar incrementalmente
   * uma instancia por uma instancia modelo.
   * 
   * @return true if the encapsulated Weka model is incremental model
   *              se o modelo Weka encapsulado e um modelo incremental
   */
  public abstract boolean isUpdateableModel();

  /**
   * Returns true if the encapsulated Weka model can produce predictions in a
   * batch.
   * 
   * PT-BR
   * 
   * Retorna verdadeiro se o modelo Weka encapsulado pode produzior previsoes em lotes.
   * 
   * @return true if the encapsulated Weka model can produce predictions in a
   *         batch
   *              se o modelo Weka encapsulado pode produzir previsoes em lotes.
   */
  public abstract boolean isBatchPredictor();

  /**
   * Update (if possible) a model with the supplied Instance
   * 
   * PT-BR
   * 
   * Atualiza (se possivel) o modelo com a Instance fornecida
   * 
   * @param inst the Instance to update the model with
   *             a Instance para atualizar o modelo  
   * @return true if the model was updated
   *              se o modelo foi atualizado
   * @exception Exception if an error occurs
   *                      se ocorrer um erro
   */
  public abstract boolean update(Instance inst) throws Exception;

  /**
   * Static factory method to create an instance of an appropriate subclass of
   * WekaScoringModel given a Weka model.
   * 
   * PT-BR
   * 
   * Metodo statico para criar uma instancia para uma subclasse adequeada para 
   * WekaScoringModel dado um modelo Weka.
   * 
   * @param model a Weka model
   *              uma modelo Weka
   * @return an appropriate WekaScoringModel for this type of Weka model
   *         uma adequada WekaScoringModel para este tipo do modelo Weka   
   * @exception Exception if an error occurs
   *                      se ocorrer um erro
   */
  public static WekaScoringModel createScorer(Object model) throws Exception {
    if (model instanceof Classifier) {
      return new WekaScoringClassifier(model);
    } else if (model instanceof Clusterer) {
      return new WekaScoringClusterer(model);
    }
    return null;
  }
}
