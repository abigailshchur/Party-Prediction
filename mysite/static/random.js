// Seeded random from http://indiegamr.com/generate-repeatable-random-numbers-in-js/
Math.seed = 6;
 
// in order to work 'Math.seed' must NOT be undefined,
// so in any case, you HAVE to provide a Math.seed
Math.seededRandom = function() {
  Math.seed = (Math.seed * 9301 + 49297) % 233280;
  var rnd = Math.seed / 233280;
  
  return rnd;//min + rnd * (max - min);
}

