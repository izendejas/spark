/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.feat

import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

// TODO: design needs work
private[feat] trait FeatureEval {

  def rank(data: RDD[LabeledPoint]): Array[(Int, Double)]

  def select(data: RDD[LabeledPoint], k: Int): Array[(Int, Double)]

}

class InfoGainFeatureEval extends FeatureEval with Logging {
  private def pLogP(prob: Double): Double = prob * math.log(prob)

  def rank(data: RDD[LabeledPoint]): Array[(Int, Double)] = {
    data.cache()  // make sure data is cached

    // TODO: consider computing label priors and joint probs in one pass like in NaiveBayes

    // first compute class priors
    val labelCounts = data.map{point => point.label -> 1L}.countByKey()
    val numInstances = labelCounts.foldLeft(0L){case (curCount, (_, count)) => curCount + count}
    val labelEntropy: Double = labelCounts.foldLeft(0D){case (curEntropy, (label, count)) =>
      curEntropy - pLogP(count / numInstances)
    }
    val brLabelCounts = data.sparkContext.broadcast(labelCounts)

    // group class with feature and compute joint probabilities
    val jointCounts = data.flatMap{point =>
      point.features.toArray.zipWithIndex.map{case (value, idx) => (idx,point.label) -> value}
    }.reduceByKey(_ + _)

    val jointCountsByFeatId = jointCounts.map{case ((featId, label), count) =>
      featId -> (label, count)
    }.groupByKey()  // TODO: define number of partitions?

    val featInfoGain = jointCountsByFeatId.map{case (featId, labelCounts) =>
      val lblCounts = brLabelCounts.getValue()

      val totalFeatCount = labelCounts.foldLeft(0D){case (total, (_, count)) => total + count}
      val featureProbability = totalFeatCount.toDouble / numInstances
      val labelGivenFeatEntropy = labelCounts.foldLeft(0D){case (curEntropy,(label, count)) =>
        curEntropy + pLogP(count / totalFeatCount)
      }
      val labelGivenNoFeatEntropy = labelCounts.foldLeft(0D){case (curEntropy, (label, count)) =>
        val negCount = lblCounts.get(label).get - count // label counts cannot be zero
        curEntropy + pLogP(negCount / (numInstances - totalFeatCount))
      }

      val infoGain = labelEntropy + featureProbability * labelGivenFeatEntropy +
        (1D - featureProbability) * labelGivenNoFeatEntropy
      featId -> infoGain
    }.collect()

    featInfoGain.sortBy(-_._2)
  }

  def select(data: RDD[LabeledPoint], k: Int): Array[(Int, Double)] = {
    rank(data).take(k)
  }

}