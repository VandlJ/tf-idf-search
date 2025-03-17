#!/usr/bin/perl

# Slouzi k vypoctu, na kolik ruznych stemu se prevedou vsechny varianty jednoho
# slova.
# Postupne prochazi vsechna slova ve vysledcich a pokud se aktualni slovo
# lisi od predchoziho v prvnich N znacich (kde N == $delkaShody), pak se jedna
# o nove slovo (a ne o variantu slova predchoziho). Pokud tedy ve vysledcich
# budou dve po sobe nasledujici slova zacinat ruznym pismenem, staci hodnotu
# promenne $delkaShody nastavit na 1. Viz napr. tento priklad:
#   sochaø
#   sochaøùv
#   zubaø
#   zubaøi
#   traktorista
#   ...
#
# vstup: soubor s vysledky stemmovani

my $progname = $0;

if (scalar @ARGV != 1) {
  print "Usage: $progname <soubor s vysledky>\n";
  exit 1;
}
$file = shift(@ARGV);
@poctyTvaru = ();   #pole s pocty ruznych stemu pro vsechny varianty daneho slova,
%stemy = ();        #hash do ktereho se ukladaji stemy vsech variant jednoho slova
%pocty = ();
$line = "";
$zadani = "";
$vysledek = "";
$prevZadani = "";
$pocetSlov = 0;     #pocet ruznych slov ve vysledcich
$delkaShody = 1;

open(DESCFILE, $file) or die "Nelze otevrit soubor.";

while($line = <DESCFILE>) {
  $line =~ s/\s+$//;      #odstraneni konce radku
  next if $line eq "";
  next if $line =~ /^#/;  #preskocit komentare

  
  $line =~ /^(.*?)\ +(.*?)$/; #zadani a vysledek jsou od sebe oddeleny mezerami
  $vysledek = $2;             #vysledek je uveden az za mezerami
  #cast zadani, na jejuz zaklade se porovnava, jestli se preslo na dalsi slovo
  $zadani = substr $1, 0, $delkaShody;

  if($zadani ne $prevZadani) { #preslo se na nove slovo
    $prevZadani = $zadani;
    if(scalar keys %stemy ne 0) {
      #pocet ruznych stemu slova s indexem cislo $pocetSlov
      @poctyTvaru[$pocetSlov] = scalar keys %stemy;
    }
    $pocetSlov++;                 #preslo se na nove slovo
    @poctyTvaru[$pocetSlov] = 0;  #vynulovani poctu ruznych stemu daneho slova
    %stemy = ();                  #vymazani seznamu ruznych stemu daneho slova
  }

  $stemy{$vysledek} = "loremipsum"; #inicializace hashe nejakym nesmyslnym textem
}
@poctyTvaru[$pocetSlov] = scalar keys %stemy; #pridani posledniho vysledku


foreach $item (@poctyTvaru) {
  next if $item eq "";
  $pocty{$item} += 1;
}

while (($key, $value) = each(%pocty))
{
  $proc = ($pocty{$key}*100) / $pocetSlov; #vypocet procent
  #tiskne vysledky ve forme citelne pro normalni lidi:
  #print "na $key ruznych stemu prevedeno $pocty{$key} slov z $pocetSlov, coz je $proc %\n";
  
  #tiskne vysledky vhodne pro vytvoreni grafu:
  #pocet ruznych stemu <tab> kolika procent pripadu se to tyka
  printf("%d\t%.1f%%\n", $key, $proc)
  
}

